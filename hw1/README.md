# HW 1
Author: Pablo Ortega-Kral (portegak)

# Q1

- For getting base mean and standar deviation I run the BC training script and read the metrics for the train set.

```
python rob831/scripts/run_hw1.py \--expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \--video_log_freq -1
```

- For training basic BC, I choose parameters eval_epoch_size = 5000, againt_step_per_ter 3500

```
python rob831/scripts/run_hw1.py \--expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \--video_log_freq -1 --eval_batch_size 5000 --num_agent_train_steps_per_iter 3500
```

```
python rob831/scripts/run_hw1.py \--expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \--video_log_freq -1 --eval_batch_size 5000 --num_agent_train_steps_per_iter 3500
```

- Experiment with hyper parameters. I vary the number of training iterations by a step of 500. For simplicity I include the command for the last number of iterations.

```
python rob831/scripts/run_hw1.py \--expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \--video_log_freq -1 --eval_batch_size 5000 --num_agent_train_steps_per_iter 7000
```


# Q2

- To get the results for training the policy with DAgger, I run the following commands. Note that I modified the script to output the iteratio vs policy results graphs. 

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --num_agent_train_steps_per_iter 3500
```

```
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 10 --do_dagger --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --video_log_freq -1 --eval_batch_size 5000 --num_agent_train_steps_per_iter 3500
```