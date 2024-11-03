import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns

def extract_scalar_events(logdir, tag):
    scalar_events = []
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    for event in ea.Scalars(tag):
        scalar_events.append((event.step, event.value))
    return scalar_events

def get_exp_tag(filename):
    tags = filename.split('_')
    seed = [int(s) for s in tags if s.isdigit()][0]
    if 'dqn' in tags:
        return 'DQN', seed
    elif 'doubledqn' in tags:
        return 'DoubleQ', seed
    else:
        return breakpoint()

def read_logs(data_dir, data_tag='Eval_AverageReturn'):
    data = extract_scalar_events(data_dir, data_tag)
    steps, values = zip(*data)
    tag, seed = get_exp_tag(data_dir)
    exp_id = f"{tag}_seed_{seed}"
    df = pd.DataFrame({'step': steps, 'value': values, 'tag': exp_id, 'network_type': f"{tag}_{data_tag}"})
    return df

def logs_to_dataframe(logs):
    data = {'log': logs}
    df = pd.DataFrame(data)
    return df


def plot(exp_df):
    import matplotlib.pyplot as plt
    avg_df = exp_df.drop(columns=['tag'])
    grouped = avg_df.groupby(['step', 'network_type'])
    avg_df = grouped.mean().reset_index()
    variance = grouped["value"].var().reset_index(drop=True)
    avg_df['std_dev'] = variance ** 0.5
    sns.lineplot(data=avg_df, x='step', y='value', hue='network_type')
    for network_type in avg_df['network_type'].unique():
        network_df = avg_df[avg_df['network_type'] == network_type]
        plt.fill_between(network_df['step'], network_df['value'] - network_df['std_dev'], network_df['value'] + network_df['std_dev'], alpha=0.2)
        # plt.errorbar(network_df['step'], network_df['value'], yerr=network_df['std_dev'], fmt='o', capsize=5, alpha=0.5)
    plt.xlabel('Step')
    plt.ylabel('Average Return (across seeds)')
    plt.title('Average Return vs Step')
    plt.legend(title='Network Type')
    plt.show()

if __name__ == "__main__":
    logdir = 'data'
    data_dirs = [os.path.join(logdir, folder) for folder in os.listdir(logdir) if 'q1' in folder]
    dfs = []
    for data_dir in data_dirs:
        df = read_logs(data_dir, 'Train_AverageReturn')
        best_df = read_logs(data_dir, 'Train_BestReturn')
        dfs.append(df)
        dfs.append(best_df)
    all_exp_df = pd.concat(dfs, ignore_index=True)
    plot(all_exp_df)
