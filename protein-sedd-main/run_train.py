import datetime # 用于超时设置（如分布式训练的超时时间）。
import os
import os.path
import gc
from itertools import chain # 用于将多个迭代器连接在一起（如参数列表）。

import numpy as np
import torch
import torch.distributed as dist # 用于 分布式训练，DDP 让多个 GPU 进行并行训练。
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage # 指数移动平均（EMA），用于稳定训练。
from transformers import GPT2TokenizerFast, GPT2LMHeadModel # Hugging Face 的 GPT2 相关工具（用于文本处理或语言建模）。


torch.backends.cudnn.benchmark = True # 启用 CuDNN 的自动优化，可以加速卷积操作（适用于固定大小的输入）。
# torch.autograd.set_detect_anomaly(True) # 如果启用，会检测反向传播中的异常（如 NaN），但会 影响训练速度。



def setup(rank, world_size, port): # rank：当前 GPU/进程编号。 world_size：总共的进程数（即 GPU 数量）。 port：通信端口（多 GPU 需要通信）。
    os.environ["MASTER_ADDR"] = "localhost" # 设定主进程的地址（在单机上通常是 localhost）。
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    ) # 使用 NCCL（NVIDIA 的高效分布式计算库） 进行进程组初始化。


def cleanup(): # 销毁分布式进程
    dist.destroy_process_group() # 释放分布式进程组，防止资源泄漏。


def run_multiprocess(rank, world_size, cfg, port): # 运行分布式训练
    try:
        setup(rank, world_size, port) # 先调用 setup() 初始化进程组。
        _run(rank, world_size, cfg) # 运行 _run()（核心训练逻辑）。
    finally:
        cleanup() # 训练完成后调用 cleanup() 释放资源。



def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank) # 设置 当前进程 绑定到 rank 对应的 GPU。
    work_dir = cfg.work_dir # 获取 工作目录（存储日志、检查点等）。

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples") # 存储生成的样本。
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth") # 存储 模型检查点（用于恢复训练）。
    if rank == 0: # 只让 主进程 创建目录，避免冲突。
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs")) # logger 负责记录训练信息。
    def mprint(msg): # 只有 rank == 0 的进程会输出日志。
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu") # 获取计算设备信息
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.") # 检查 GPU 数量和显存大小，如果 没有 GPU，则会警告并改用 CPU。

    # build token graph
    graph = graph_lib.get_graph(cfg, device) # 获取 图结构
    
    # build score model
    score_model = SEDD(cfg).to(device) # 加载 SEDD 模型。
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True) # 使用 分布式数据并行（DDP） 进行 多 GPU 训练。

    num_parameters = sum(p.numel() for p in score_model.parameters()) # 统计模型参数数量 score_model.parameters() 返回模型所有参数的迭代器。p.numel() 计算每个参数张量中的元素总数（即参数个数）。sum(...) 计算整个模型的参数总数，并打印出来。
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}") # 初始化指数移动平均（EMA）ExponentialMovingAverage 通过 decay 参数（一般取值接近 1，如 0.999）计算指数加权平均。这个方法主要用于平滑模型参数，使其对近期更新的参数更加敏感，有助于提高模型的稳定性。

    # build noise
    noise = noise_lib.get_noise(cfg).to(device) # noise_lib.get_noise(cfg) 获取一个符合配置的噪声模型。 noise.to(device) 将其移动到 GPU 设备上。
    noise = DDP(noise, device_ids=[rank], static_graph=True) # DDP(...) 采用 DistributedDataParallel（DDP）方式并行训练。
    sampling_eps = 1e-5 # sampling_eps = 1e-5 设定一个很小的采样误差阈值。


    # build optimization state 设置优化器与混合精度计算
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters())) # losses.get_optimizer(cfg, chain(...)) 获取优化器，优化目标包括 score_model 和 noise 两部分参数。
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler() # GradScaler() 用于混合精度训练（AMP），减少显存占用，提高计算效率。
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0)  # state 变量存储训练状态，包括：optimizer（优化器）scaler（混合精度缩放器）model（主模型）noise（噪声模型）ema（指数移动平均）step（当前训练步数）


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device) # 从指定路径恢复之前的训练状态（如果有的话）。
    initial_step = int(state['step']) # 记录恢复后的训练步数。

    
    # original tokenizer:
    #tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # protein tokenizer 
    #tokenizer = data.CharacterTokenizer() 

    from collections import OrderedDict
    from transformers import GPT2TokenizerFast
    import json

    # Define amino acids and special tokens 初始化自定义 GPT-2 分词器 定义氨基酸和特殊 token
    amino_acids = list("RSHVE")
    #special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    #all_tokens = special_tokens + amino_acids
    all_tokens = amino_acids

    # Create the vocabulary 创建词汇表
    vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))

    # Save the vocabulary 保存词汇表
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)

    # Create an empty merges.txt file 创建 merges.txt（BPE 规则文件）
    with open('merges.txt', 'w') as f:
        f.write('#version: 0.2\n')

    # Initialize the tokenizer 初始化分词器 读取这两个文件，初始化自定义 GPT-2 分词器。
    tokenizer = GPT2TokenizerFast(
        vocab_file='vocab.json',
        merges_file='merges.txt',
        #bos_token='<s>',
        #eos_token='</s>',
        #unk_token='<unk>',
        #pad_token='<pad>',
        #mask_token = '<mask>'
    )


    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg) # 加载训练和评估数据集。

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds) #通过 iter(...) 将数据集转换为可迭代对象。
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg) # 负责管理优化过程（如梯度裁剪）。
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum) # 获取训练步骤函数，True 表示用于训练。
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum) # 获取评估步骤函数，False 表示用于验证。


    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    '''训练循环 训练 num_train_steps 轮，加载 batch 数据后调用 train_step_fn 计算损失 loss。'''
    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']


        if cfg.data.train != "text8":
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)

        #print(torch.argmax(batch))
        #print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}, device: {batch.device}")
        #print(f"Batch min: {batch.min().item()}, max: {batch.max().item()}")

        

        loss = train_step_fn(state, batch)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']: # 说明本轮训练完成
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item())) # 每 log_freq 轮打印训练损失。
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                if cfg.data.valid != "text8":
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                else:
                    eval_batch = next(train_iter).to(device)
                eval_loss = eval_step_fn(state, eval_batch)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item())) # 每 eval_freq 轮计算评估损失 eval_loss。

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state) # 每 snapshot_freq 轮保存模型检查点。

                # Generate and save samples 生成样本 使用 sampling_fn(score_model) 生成样本，并解码成文本保存到 sample_{rank}.txt。
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                    sentences = tokenizer.batch_decode(sample)
                    
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("============================================================================================\n")

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            pass
                            # Let's think about how to evaluate this 

#                             eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
#                             batches = sample.shape[0] // cfg.eval.perplexity_batch_size
#                             total_perplexity = 0
#                             for i in range(batches):
#                                 s = sample[i * cfg.eval.perplexity_batch_size:(i + 1) * cfg.eval.perplexity_batch_size]
#                                 print(s)
#                                 loss, logits = eval_model(s, labels=s)[:2]
#                                 logits = logits.transpose(-1, -2)
#                                 perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
#                                 total_perplexity += perplexity
#                             total_perplexity /= batches
#                             dist.all_reduce(total_perplexity)
#                             total_perplexity /= world_size
#                             mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")

#                             del eval_model, logits, loss

                    dist.barrier()
