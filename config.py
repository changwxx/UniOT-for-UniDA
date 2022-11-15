import yaml
import easydict
import os
import datetime

import argparse
parser = argparse.ArgumentParser(description='Code for *UniOT for UniDA*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu_index', type=str, default='0', help='')
parser.add_argument('--exp', type=str, default='debug', help='')
parser.add_argument('--dataset', type=str, default='office31', help='dataset')
parser.add_argument('--source', type=str, default='amazon', help='source domain')
parser.add_argument('--target', type=str, default='dslr', help='target domain')
parser.add_argument('--model_path', type=str, default=None, help='/path/to/your/model/final.pkl')

parser_args = parser.parse_args()

config_file = f'config/{parser_args.dataset}-config.yaml'
save_config = yaml.safe_load(open(config_file))
args = easydict.EasyDict(save_config)

# assign args to variables
if parser_args.source is None:
    source = args.data.dataset.source_default
else:
    source = parser_args.source
if parser_args.target is None:
    target = args.data.dataset.target_default
else:
    target = parser_args.target

# fill in <xxx> in config
dataset_root = args.data.dataset.root_path

current_dir = os.path.dirname(__file__)

source_txt = args.data.dataset.s_txt_path.replace("<source>", source)
source_txt = source_txt.replace("<current_dir>", current_dir)
target_txt = args.data.dataset.t_txt_path.replace("<target>", target)
target_txt = target_txt.replace("<current_dir>", current_dir)

pretrained_model_path = args.model.pretrained_model_path.replace("<current_dir>", current_dir)

log_path = args.log.root_dir.replace("<exp>", parser_args.exp)
transfer_task = f'{source}2{target}'
log_path = log_path.replace("<transfer_task>", transfer_task)
now = datetime.datetime.now().strftime('%b%d_%H-%M')
log_path = log_path.replace("<now>", now)
log_path = log_path.replace("<current_dir>", current_dir)

# dataloader paramter
batch_size = args.data.dataloader.batch_size
num_workers = args.data.dataloader.data_workers

# hyper-parameters
K = args.param.K
gamma = args.param.gamma
mu = args.param.mu
temp = args.param.temp
lam = args.param.lam
MQ_size = args.param.MQ_size
