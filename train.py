import argparse
import yaml
from train_main import train_main
import random
import numpy as np
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from datetime import datetime

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    parser = argparse.ArgumentParser(description='train a model with your params')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    print('.........................Parsing ......')
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_seed = random.randint(0, 1000000)
    seed_everything(random_seed)
    train_main(config, random_seed,run_time)


if __name__ == '__main__':
    main()
    
    


