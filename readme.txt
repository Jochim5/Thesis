use:  python run_sample.py --model_path MODEL_PATH  togenerate robot from DDM
in examples/run_ga.py, you can choose to use 1)original GA
                                                                                    2)modified GA that allows you to start from a spesific robot, you need to Change the robot path on row 98 in examples/ga/run_hand.py
                                                                                    3)fast GA you also Need to Change the template path and the robot path on row 98,99 in  examples/comparison/run
These 3 Option is according to the last 3 rows, you Need to select one. you can also directly change exp_name, env_name, pop_size, structure_shape, max_evaluations, num_cores in run_ga.py
you can calculate similarity reward in examples/comparison/compare_robot.py you need to change the 25 letter string  or the sturcture Matrix
you can run  run_ppo.py to get optimal Control law just Change the robot Name

youcan find the DDM here"https://drive.google.com/drive/folders/1u-U5PFAJCK79j2kkuihnlN3i9xos03g-?usp=sharing"
create a protein-sedd-main\exp_local folder and put the 2025.03.09 into it
put custom_data.txt and merges.txt in to protein-sedd-main

