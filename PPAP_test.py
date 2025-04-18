from strips.PPAP import *
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd


checkpoint_path = './weight/epoch=11-val_r2=0.361.ckpt'
model = PPAP.load_from_checkpoint(
    checkpoint_path,
    d_node=2560,
    n_heads=20,
    num_layers=1,
    lr=1e-4
)
model.to('cuda')


def concatenate_tensors_by_ids(id_list, directory):
    """
    根据提供的id列表，从指定目录加载对应的.pt文件，并将它们按照列表顺序拼接成一个大张量。

    参数:
        id_list (list): 包含id的列表。
        directory (str): 保存.pt文件的目录。

    返回:
        torch.Tensor: 拼接后的张量。
    """
    tensors = []  # 用于存储加载的张量
    for id_ in id_list:
        # 构造文件路径
        file_path = f"{directory}/{id_}.pt"

        # 加载张量
        tensor = torch.load(file_path)

        # 将张量添加到列表中
        tensors.append(tensor)

    # 按顺序拼接所有张量
    concatenated_tensor = torch.cat(tensors, dim=0)

    return concatenated_tensor


class HDF5Dataset(Dataset):
    def __init__(self, file_path, node_path, dataset_ids=None):
        self.file_path = file_path
        self.node_path = node_path
        self.hdf5_file = h5py.File(file_path, 'r')
        self.all_keys = list(self.hdf5_file.keys())

        # train_ids是一个包含训练样本的ID的列表
        # 如果train_ids被提供，那么只包含ID在train_ids中的样本
        if dataset_ids is not None:
            self.keys = [key for key in self.all_keys if key in dataset_ids]
        else:
            self.keys = self.all_keys

    def __len__(self):
        # 返回数据集中的样本个数
        return len(self.keys)

    def __getitem__(self, index):
        # 根据索引获取hdf5文件中的数据
        group_key = self.keys[index]
        group = self.hdf5_file[group_key]

        # 将数据从hdf5提取出来，并转换成tensor
        core_edge_fea = torch.from_numpy(np.array(group['core_edge_fea'])).float()
        inner_edge_fea = torch.from_numpy(np.array(group['inner_edge_fea'])).float()
        # edge_index = torch.from_numpy(np.array(group['edge_index'])).long()
        inner_edge_index = torch.from_numpy(np.array(group['inner_edge_index'])).long()
        core_edge_index = torch.from_numpy(np.array(group['core_edge_index'])).long()
        affinity = torch.from_numpy(np.array(group['affinity'])).float()
        batch_index = torch.from_numpy(np.array(group['batch_index'])).long()
        chain = group['chain'][()].decode('utf-8')  # 这是一个字符串，不是数字型，无需转为Tensor

        chain_list = []
        for i in chain:
            chain_list.append(f'{group_key}_{i}')

        node_fea = concatenate_tensors_by_ids(chain_list, self.node_path)

        # 将数据和标签包装为dict返回
        sample = {
            'group_key': group_key,
            'core_edge_fea': core_edge_fea,
            'inner_edge_fea': inner_edge_fea,
            'inner_edge_index': inner_edge_index,
            'core_edge_index': core_edge_index,
            'affinity': affinity,
            'batch_index': batch_index,
            'node_fea': node_fea
        }

        return sample

    def close(self):
        # 确保在数据集不再使用时关闭hdf5文件
        self.hdf5_file.close()


def get_names(test_txt):
    file_name = []

    with open(test_txt, 'r') as f:
        lines = f.readlines()
        lines = [i.strip() for i in lines]
        for line in lines:
            name = line.split('_')[:-2]
            name = '_'.join(name)
            file_name.append(name)
    return file_name


def delta_G_to_KD_with_units(delta_G, T):
    R = 0.0019872041  # kcal/(mol·K)
    KD = math.exp(-delta_G / (R * T))

    # Determine the appropriate unit based on the KD value
    if KD >= 1e-3:
        unit = "M"
    elif KD >= 1e-6:
        unit = "μM"
    elif KD >= 1e-9:
        unit = "nM"
    elif KD >= 1e-12:
        unit = "pM"
    else:
        unit = "fM"

    # Convert KD to the appropriate unit
    if unit == "μM":
        KD *= 1e6
    elif unit == "nM":
        KD *= 1e9
    elif unit == "pM":
        KD *= 1e12
    elif unit == "fM":
        KD *= 1e15

    return f"{KD:.3f} {unit}"


file_path = './graph/whole_edge_feature.hdf5'
node_path = './graph/esm_feature'
test_txt = './input/for_test.txt'  # pdbbind90_test or skempi_125+2 or testset
test_name = get_names(test_txt=test_txt)
test_dataset = HDF5Dataset(file_path, node_path, dataset_ids=test_name)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
excel_path = './output/result.xlsx'


# 模型预测
model.eval()
with torch.no_grad():
    out_put = []
    for batch in test_loader:
        group_key = batch['group_key'][0]  # 获取group_key
        node_feat = batch['node_fea'].cuda()
        inner_edge_feat = batch['inner_edge_fea'].cuda()
        core_edge_fea = batch['core_edge_fea'].cuda()
        inner_edge_idx, core_edge_idx, batch_idx = batch['inner_edge_index'].cuda(), batch['core_edge_index'].cuda(), batch['batch_index'].cuda()
        prediction, att = model(node_feat, core_edge_fea, core_edge_idx)
        prediction_value = round(prediction.item(), 3)
        T = 25 + 273.15  # Convert 25 degrees Celsius to Kelvin
        kd = delta_G_to_KD_with_units(prediction_value, T)
        out_put.append([group_key, prediction_value, kd])
        print(f'finish {group_key} ')

    df = pd.DataFrame(out_put, columns=['name', '-ΔG(kcal/mol)', 'Kd'])
    df.to_excel(excel_path, index=False)

