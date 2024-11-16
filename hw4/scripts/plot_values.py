import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns

def extract_substring(logdir, setting="reacher", environment="ensemble", question="q4"):
    import re

    # Define the regex pattern dynamically using the environment parameter
    if environment == "ensemble":
        pattern = fr"{question}_{setting}_{environment}\d+"
    else: 
        pattern = fr"{question}_{setting}[^/]*hw4"
    match = re.search(pattern, logdir)
    
    # Return the matched substring or an empty string if not found
    return match.group(0) if match else ""

def extract_scalar_events(logdir, tag):
    scalar_events = []
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    for event in ea.Scalars(tag):
        scalar_events.append((event.step, event.value))
    return scalar_events

def plot_train_eval_returns(logdir, title='Returns for Q2 Obstacles'):
    import matplotlib.pyplot as plt
    train_tag = 'Eval_AverageReturn'
    eval_tag = 'Train_AverageReturn'

    train_events = extract_scalar_events(logdir, train_tag)
    eval_events = extract_scalar_events(logdir, eval_tag)

    train_df = pd.DataFrame(train_events, columns=['step', 'Train_AverageReturn'])
    eval_df = pd.DataFrame(eval_events, columns=['step', 'Eval_AverageReturn'])

    merged_df = pd.merge(train_df, eval_df, on='step')

    sns.scatterplot(data=merged_df, x='step', y='Train_AverageReturn', label='Eval Average Return')
    sns.scatterplot(data=merged_df, x='step', y='Eval_AverageReturn', label='Train Average Return')
    sns.set(style='whitegrid')
    sns.despine()
    plt.title(title)
    plt.savefig('train_eval_returns.png', dpi=500)
    plt.show()

def plot_Q3(DPI=600):
    import matplotlib.pyplot as plt
    tag = 'Eval_AverageReturn'

    # Obstacles
    logdir = 'rob831/data/hw4_q3_obstacles_obstacles-hw4_part1-v0_15-11-2024_19-34-59'
    q3_events = extract_scalar_events(logdir, tag)
    q3_df = pd.DataFrame(q3_events, columns=['step', 'Eval_AverageReturn'])
    sns.lineplot(data=q3_df, x='step', y='Eval_AverageReturn')
    sns.set(style='whitegrid')
    sns.despine()
    plt.title("Q3 Obstacles Returns")
    plt.savefig('q3_obstacles.png', dpi=DPI)
    plt.show()

    # Reacher
    logdir = 'rob831/data/hw4_q3_reacher_reacher-hw4_part1-v0_15-11-2024_19-39-27'
    q3_events = extract_scalar_events(logdir, tag)
    q3_df = pd.DataFrame(q3_events, columns=['step', 'Eval_AverageReturn'])
    sns.lineplot(data=q3_df, x='step', y='Eval_AverageReturn')
    sns.set(style='whitegrid')
    sns.despine()
    plt.title("Q3 Reacher Returns")
    plt.savefig('q3_reacher.png', dpi=DPI)
    plt.show()

    # HalfCheetah
    logdir = 'rob831/data/hw4_q3_cheetah_cheetah-hw4_part1-v0_15-11-2024_20-04-36'
    q3_events = extract_scalar_events(logdir, tag)
    q3_df = pd.DataFrame(q3_events, columns=['step', 'Eval_AverageReturn'])
    sns.lineplot(data=q3_df, x='step', y='Eval_AverageReturn')
    sns.set(style='whitegrid')
    sns.despine()
    plt.title("Q3 Cheetah Returns")
    plt.savefig('q3_cheetah.png', dpi=DPI)

def plot_Q4(DPI=600):
    import matplotlib.pyplot as plt
    tag = 'Eval_AverageReturn'

    logdir_base = 'rob831/data/'
    q4_dirs = [os.path.join(logdir_base, d) for d in os.listdir(logdir_base) if 'hw4_q4' in d]

    ensemble_sweep = [d for d in q4_dirs if 'ensemble' in d]
    horizon_sweep = [d for d in q4_dirs if 'horizon' in d]
    num_seq = [d for d in q4_dirs if 'numseq' in d]

    # Ensemble
    ensemble_data = []
    for logdir in ensemble_sweep:
        q4_events = extract_scalar_events(logdir, tag)
        for step, value in q4_events:
            abrev_tag = extract_substring(logdir, environment="ensemble")
            ensemble_data.append((abrev_tag, step, value))
    ensemble_df = pd.DataFrame(ensemble_data, columns=['Experiment', 'step', 'Eval_AverageReturn'])
    sns.lineplot(data=ensemble_df, x='step', y='Eval_AverageReturn', hue='Experiment')
    sns.set(style='whitegrid')
    sns.despine()
    plt.title("Q4 Ensemble Returns")
    plt.savefig('q4_ensemble.png', dpi=DPI)
    plt.show()

    # Horizon
    horizon_data = []
    for logdir in horizon_sweep:
        q4_events = extract_scalar_events(logdir, tag)
        for step, value in q4_events:
            abrev_tag = extract_substring(logdir, environment="horizon")
            horizon_data.append((abrev_tag, step, value))
    horizon_df = pd.DataFrame(horizon_data, columns=['Experiment', 'step', 'Eval_AverageReturn'])
    sns.lineplot(data=horizon_df, x='step', y='Eval_AverageReturn', hue='Experiment')
    sns.set(style='whitegrid')
    sns.despine()
    plt.title("Q4 Horizon Returns")
    plt.savefig('q4_horizon.png', dpi=DPI)
    plt.show()

    # Num Seq
    numseq_data = []
    for logdir in num_seq:
        q4_events = extract_scalar_events(logdir, tag)
        for step, value in q4_events:
            abrev_tag = extract_substring(logdir, environment="numseq")
            numseq_data.append((abrev_tag, step, value))
    numseq_df = pd.DataFrame(numseq_data, columns=['Experiment', 'step', 'Eval_AverageReturn'])
    sns.lineplot(data=numseq_df, x='step', y='Eval_AverageReturn', hue='Experiment')
    sns.set(style='whitegrid')
    sns.despine()
    plt.title("Q4 Num Seq Returns")
    plt.savefig('q4_numseq.png', dpi=DPI)
    plt.show()

def plot_Q5():
    import matplotlib.pyplot as plt
    tag = 'Eval_AverageReturn'

    logdir_base = 'rob831/data/'
    q5_dirs = [os.path.join(logdir_base, d) for d in os.listdir(logdir_base) if 'hw4_q5' in d]

    q5_data = []
    for logdir in q5_dirs:
        q5_events = extract_scalar_events(logdir, tag)
        for step, value in q5_events:
            abrev_tag = extract_substring(logdir, question="q5", setting="cheetah", environment="")
            q5_data.append((abrev_tag, step, value))
    q5_df = pd.DataFrame(q5_data, columns=['Experiment', 'step', 'Eval_AverageReturn'])
    sns.lineplot(data=q5_df, x='step', y='Eval_AverageReturn', hue='Experiment')
    sns.set(style='whitegrid')
    sns.despine()
    plt.title("Q5 Returns")
    plt.savefig('q5_returns.png', dpi=600)
    plt.show()

if __name__ == '__main__':

    plot_Q5()