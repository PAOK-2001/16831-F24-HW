import os
import re
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

out_dir = 'results/'

def parse_string(s):
    # Regular expression to match the parameters
    pattern = r"b(\d+)_lr([0-9.]+)"
    learning_params = re.search(pattern, s)
    # Init tags
    batch_size = ""
    lr = ""
    nn_baseline = ""
    rtg = ""

    if learning_params:
        batch_size = learning_params.group(1)
        lr = learning_params.group(2)

    if "--nn_bassline" in s:
        nn_baseline = "_nn_baseline"
    if "rtg" in s:
        rtg = "_rtg"

    if nn_baseline:
        nn_baseline = nn_baseline.group(1)
    return f"b{batch_size}_lr{lr}{nn_baseline}{rtg}"

questions = {
    # "Small Batch Experiments": "q1_sb",
    # "Large Batch Experiments": "q1_lb",
    # "Inverted Pendulum": "q2_b100_r0.02_rtg_InvertedPendulum-v4_04-10-2024_02-52-47",
    # "Lunar Lander": "q3_b10000_r0.005_LunarLanderContinuous-v2_04-10-2024_23-23-40",
    # "Half Cheetah Experiments": "q4_search",
    # "Half Cheetah Hyperparameter Sweep": "q7_",
    "Optimal Hyperparameters": "q7_optimal",
    # "Generalized Advantage Estimation": "q5_",
}

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
            if "q7" in experiment_tag:
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

    for exp, question in questions.items():
        # Find all dirs with prefix question
        logdirs = [os.path.join(logdir, d) for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d)) and question in d]
        
        tag = 'Eval_AverageReturn'
        output_file = f'{out_dir}/{exp}.png'
        title = exp
        plot_graph(logdirs, tag, output_file, title)
