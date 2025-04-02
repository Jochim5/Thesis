import re # 用于正则表达式处理字符串。
import os
from transformers import GPT2TokenizerFast # GPT-2 的快速分词器，用于 token 化文本数据。
from datasets import load_dataset # Hugging Face datasets 库，用于加载数据集。
from itertools import chain # itertools.chain 用于合并多个列表。
import numpy as np
import tempfile # 用于创建临时文件。
#from Bio import SeqIO
import requests # 用于进行 HTTP 请求，下载数据。
import json # 用于处理 JSON 数据。
from datasets import Dataset # Hugging Face datasets 库中的 Dataset 类，用于存储数据。
from torch.utils.data import DataLoader, DistributedSampler # PyTorch 数据加载器。 用于分布式训练时的数据采样。

#base_path = "C:\\d_pan\\PythonProject\\pythonProject\\pythonProject\\protein-sedd-main"
base_path = "/mnt/c/d_pan/PythonProject/pythonProject/pythonProject/protein-sedd-main"


def cycle_loader(dataloader, sampler=None): # 这是一个无限循环的数据加载器。
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000)) # 如果使用了 DistributedSampler，则在每个 epoch 之前设置一个随机种子 (sampler.set_epoch(...)) 以确保数据的随机性。
        for data in dataloader:
            yield data #循环遍历 dataloader 并逐个返回数据样本，使得数据流可以无限循环。


def wt_detokenizer(string): # wt_detokenizer 主要用于去除 wikitext 数据集的标记化（tokenization）符号，使文本更接近自然语言格式。
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x): # 用于 Penn Treebank 语料。
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x): # 用于 Google One Billion Words 语料。
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text): # 处理 lambada 语料。
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()

def acyp_detokenizer(text): # 返回原文本，不做修改。
    return text 


def get_lambada_test_dataset(): # 下载 LAMBADA 数据集（用于语言模型评估）。
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url): # 读取 JSONL 格式的数据并转换为 Hugging Face 的 Dataset。
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


def get_acyp_dataset(): # 该函数用于获取 ACYP 数据集
    def read_txt_file(filepath):  # 定义读取 .txt 文件的函数
        sequence_list = []
        with open(filepath, "r") as file:  # 打开文件
            for line in file:  # 逐行读取
                line = line.strip()  # 去除行首尾的空白字符（如换行符）
                if len(line) == 25:  # 确保每行是 25 个字母
                    sequence_list.append({
                        "text": line  # 将每行数据作为 "text" 字段
                    })
        return sequence_list


    file_path = os.path.join(base_path, "custom_data.txt")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 '{file_path}' 不存在。")

    # 读取并解析 .txt 文件
    custom_data = read_txt_file(file_path)  # 调用读取函数
    dataset = Dataset.from_list(custom_data)  # 转换为 Hugging Face 的 Dataset 格式

    return dataset


'''# name: 指定数据集名称（如 "wikitext103", "ptb", "acyp"）。
mode: 指定数据集的模式（"train"、"validation"、"test"）。
cache_dir: 指定 Hugging Face 数据集的缓存目录。
block_size: 训练数据的分块大小，默认为 1024。
num_proc: 并行处理的进程数，默认为 8。'''

def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir) # load_dataset()：使用 Hugging Face datasets 库加载数据集。
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset() # get_lambada_test_dataset() 和 get_acyp_dataset()：用于特定数据集的自定义加载逻辑。
    elif name == "acyp":
        dataset = get_acyp_dataset() 
    elif name == "uniref50":
        dataset = load_dataset("agemagician/uniref50", cache_dir=cache_dir)
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    if name == "lambada": # 选择数据模式
        data = dataset
    elif name == "acyp":
        data = dataset 
    else:
        data = dataset[mode]

    if name.startswith("wikitext"): # 选择去标记化（detokenizer）
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    elif name in ["acyp", "uniref50"]:
        detokenizer = acyp_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer): #  定义去标记化函数
        def detok(text): # 遍历 text，对每个文本 t 应用 detokenizer 处理。
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') # 加载预训练的 GPT-2 分词器。
    EOS = tokenizer.encode(tokenizer.eos_token)[0] # 获取 GPT-2 终止符 eos_token 的编码。

    if name in ["acyp", "uniref50"]: # 这里对 acyp 和 uniref50 数据集使用自定义的 GPT2TokenizerFast，加载本地的 vocab.json 和 merges.txt 词表。

        # Initialize the tokenizer
        tokenizer = GPT2TokenizerFast(
            vocab_file='vocab.json',
            merges_file='merges.txt',
            #bos_token='<s>',
            #eos_token='</s>',
            #unk_token='<unk>',
            #pad_token='<pad>',
            #mask_token='<mask>'
        )
        EOS = tokenizer.encode(tokenizer.eos_token)[0]


    def preprocess_and_tokenize(example): # 预处理和分词
        if name == "ptb":
            text = example['sentence']
        if name == "acyp":
            #print(example["text"])
            text = example["text"] # example['text']：获取文本数据。
        else:
            text = example["text"]
        # print(list(example.keys()))
        # exit()
        
        if detokenizer is not None: # 如果 detokenizer 存在，则先进行去标记化处理。
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False) # 使用 GPT-2 分词器对文本进行分词。
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        for token in tokens['input_ids']:
            token.append(EOS) # token.append(EOS)：确保所有文本都以 EOS 结尾。
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True) # 应用 preprocess_and_tokenize map()：对整个数据集批量应用 preprocess_and_tokenize，并行加速（num_proc=8）。load_from_cache_file=True：避免重复处理相同的数据。
    if name == "ptb": # 移除原始文本列 由于数据已经被转换为 token ID，因此删除 text 或 sentence 列，节省存储空间。
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
    

    def group_texts(examples): # 文本分块
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()} # chain(*examples[k])：将所有 token 拼接成一个大序列。
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size # 计算最大可用长度，确保数据块的大小为 block_size（1024）。
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)] # 将数据分成 block_size 大小的块。
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True) # map(group_texts)：将 token 数据整理成 1024 长度的块。
    chunked_dataset = chunked_dataset.with_format('torch') # 使数据集可以直接用于 PyTorch 训练。

    return chunked_dataset # 生成最终的数据集。


def get_dataloaders(config, distributed=True): # 是否使用 分布式训练，默认为 True。
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0: # 确保 batch size 是 ngpus * accum 的整数倍，否则训练无法正确并行。config.ngpus：使用的 GPU 数量。config.training.accum：梯度累积步数（梯度累积用于模拟更大的 batch size）。
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length) # 加载训练和验证数据集
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid != "text8" else "test", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed: # 设置数据采样器（Distributed Sampler）
        train_sampler = DistributedSampler(train_set)  # 如果是 分布式训练，则需要 DistributedSampler 来确保数据均匀分配到每个 GPU。否则，使用 PyTorch 默认的随机采样。
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    
    '''创建 DataLoader'''
    train_loader = cycle_loader(DataLoader(
        train_set, # train_set：训练数据集。
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum), # 计算每个 GPU 上的 batch size。
        sampler=train_sampler, # 如果是分布式训练，使用 DistributedSampler，否则使用默认的采样方式。
        num_workers=4, # 使用 4 个子进程进行数据加载，加速数据读取。
        pin_memory=True, # 固定数据到 CPU 内存，提高 GPU 训练性能（适用于 torch.cuda）。
        shuffle=(train_sampler is None), # 如果没有 train_sampler（即非分布式训练），则对数据进行随机打乱。
        persistent_workers=True, # 保持 worker 进程运行，避免每次 DataLoader 重新初始化时重复创建进程，提高效率（适用于多 GPU 训练）。
    ))
    '''参数与 train_loader 类似，但用于验证数据集。 '''
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None), # shuffle=False（验证集不需要随机打乱）。
    ))
    return train_loader, valid_loader

