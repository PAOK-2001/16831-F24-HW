import os
import re
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

out_dir = 'results/'

EXP_DICT = {
    "Small Batch Experiments": [
        "q1_sb_no_rtg_dsa_CartPole-v0_04-10-2024_02-35-16",
        "q1_sb_rtg_dsa_CartPole-v0_04-10-2024_02-36-29",
        "q1_sb_rtg_na_CartPole-v0_04-10-2024_02-37-42"
    ],

    "Large Batch Experiments":[      
        "q1_lb_no_rtg_dsa_CartPole-v0_04-10-2024_03-09-59",
        "q1_lb_rtg_dsa_CartPole-v0_04-10-2024_03-15-23",
        "q1_lb_rtg_na_CartPole-v0_04-10-2024_03-20-00"
    ],

    "Inverted Pendulum": ["q2_b100_r0.02_rtg_InvertedPendulum-v4_04-10-2024_02-52-47"],
    "Lunar Lander": ["q3_b10000_r0.005_LunarLanderContinuous-v2_04-10-2024_23-23-40"],

    "Half Cheetah Experiments": [
        "q4_search_b10000_lr0.02_HalfCheetah-v4_04-10-2024_23-54-05",
        "q4_search_b10000_lr0.02_nnbaseline_HalfCheetah-v4_05-10-2024_00-18-03",
        "q4_search_b10000_lr0.02_rtg_HalfCheetah-v4_05-10-2024_00-06-54",
        "q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_05-10-2024_00-30-36"
    ],

    "Half Cheetah Hyperparameter Sweep":[
        "q4_b10000_lr0.01-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_03-53-42",
        "q4_b10000_lr0.02-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_04-40-57",
        "q4_b10000_lr0.005-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_03-06-22",
        "q4_b30000_lr0.01-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_08-55-09",
        "q4_b30000_lr0.02-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_11-13-15",
        "q4_b30000_lr0.005-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_06-37-47",
        "q4_b50000_lr0.01-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_17-46-21",
        "q4_b50000_lr0.02-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_18-40-50",
        "q4_b50000_lr0.005-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_14-40-28"
    ],

    "Optimal Hyperparameters": [
        "q4_optimal_b30000_lr0.02_HalfCheetah-v4_05-10-2024_20-09-29",
        "q4_optimal_b30000_lr0.02_rtg_HalfCheetah-v4_05-10-2024_20-47-37",
        "q4_b30000_lr0.02-rtg_nn_baseline_HalfCheetah-v4_05-10-2024_11-13-15",
        "q4_optimal_nnbaseline_HalfCheetah-v4_05-10-2024_21-20-52"
    ],

    "Generalized Advantage Estimation": [
        "q5_b2000_r0.001_baseline_Hopper-v4_04-10-2024_22-39-34",
        "q5_b2000_r0.001_lambda0_Hopper-v4_04-10-2024_22-48-09",
        "q5_b2000_r0.001_lambda0.95_Hopper-v4_04-10-2024_22-56-19",
        "q5_b2000_r0.001_lambda0.99_Hopper-v4_04-10-2024_23-05-27",
        "q5_b2000_r0.001_lambda1_Hopper-v4_04-10-2024_23-14-16",
    ],
}

def parse_string(s):
    pattern = r"b(\d+)_lr([0-9.]+)"
    learning_params = re.search(pattern, s)
    batch_size = ""
    lr = ""
    nn_baseline = ""
    rtg = ""
    if learning_params:
        batch_size = learning_params.group(1)
        lr = learning_params.group(2)

    if "_nn_baseline" in s or "nnbaseline" in s:
        nn_baseline = "_nn_baseline"
    if "rtg" in s:
        rtg = "_rtg"
    return f"b{batch_size}_lr{lr}{nn_baseline}{rtg}"


def extract_scalar_events(logdir, tag):
    scalar_events = []
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    for event in ea.Scalars(tag):
        scalar_events.append((event.step, event.value))
    return scalar_events

def plot_graph(logdirs, tag, output_file, title):
    plt.figure(figsize=(16, 10))  # Set the figure size to 16:10 aspect ratio
    for logdir in logdirs:
        experiment_tag = os.path.basename(logdir)
        scalar_events = extract_scalar_events(logdir, tag)
        if scalar_events:
            steps, values = zip(*scalar_events)
            if "q4" in experiment_tag:
                experiment_tag = parse_string(experiment_tag)
            plt.plot(steps, values, label=experiment_tag, linewidth=2.5)  # Set line width to 2.5
    
    plt.xlabel('Iterations', fontsize=14)  # Increase font size
    plt.ylabel('Average Eval Return', fontsize=14)  # Increase font size
    plt.legend(fontsize=12)  # Increase legend font size
    plt.title(title, fontsize=16)  # Increase title font size
    plt.savefig(output_file, dpi = 500)
    # plt.show()

if __name__ == "__main__":
    logdir = 'data/'
    os.makedirs(out_dir, exist_ok=True)
    for exp, logdirs in EXP_DICT.items():
        logdirs = [f"{logdir}{log}" for log in logdirs]        
        tag = 'Eval_AverageReturn'
        output_file = f'{out_dir}/{exp}.png'
        title = exp
        plot_graph(logdirs, tag, output_file, title)
