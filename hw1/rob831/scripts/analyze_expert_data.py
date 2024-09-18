import os
import pickle
import numpy as np

class DataAnalyzer:
    def __init__(self, data_dir, out):
        self.base_dir = data_dir
        self.tasks = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "Walker2d-v2"]
    
    def load_expert_data(self, data_path) -> list:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data

    def get_data_stats(self):
        for task in self.tasks:
            data_dir = os.path.join(self.base_dir, f"expert_data_{task}.pkl")
            observation_list = self. load_expert_data(data_dir)
            breakpoint()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type= str, default="rob831/expert_data")
    parser.add_argument("--out", type=str, default= "rob831/expert_data/stats")
    args = parser.parse_args()
    
    analyzer = DataAnalyzer(**vars(args))
    analyzer.get_data_stats()