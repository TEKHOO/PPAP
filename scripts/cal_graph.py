from Bio.PDB import *
import os
import shutil
import json
import torch
import h5py
import numpy as np
import esm


"""
edge feature
"""


def extract_aa_chains(input_pdb_file, output_pdb_file, chains_to_keep):
    """
    输入：pdb文件， 需要保留的链
    输出：pdb_clean文件（只有受体配体链的pdb文件）
    """

    # 创建PDB解析器对象
    parser = PDBParser(QUIET=True)

    # 从PDB文件中解析结构
    structure = parser.get_structure('pdb_structure', input_pdb_file)

    # 创建一个新的结构对象
    new_structure = structure.copy()

    # 每条链只保留氨基酸，删除非氨基酸残基、水分子、金属离子和其他小分子等
    for model in new_structure:
        for chain in list(model):
            chain_id = chain.id
            if chain_id not in chains_to_keep or chain_id == " ":
                new_structure[model.id].detach_child(chain_id)
            else:
                residues_to_remove = []
                for residue in chain:
                    if not is_aa(residue):
                        residues_to_remove.append(residue)
                for residue in residues_to_remove:
                    chain.detach_child(residue.id)

    # 写入新的PDB文件
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb_file)


def process_dataset(txt_file, pdb_path, out_path):
    chain_info = {}
    num = 0

    # 读取数据集的受体配体链信息
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line_list = line.split('_')
            name = line.split('_')[:-2]
            name = '_'.join(name)
            chain_remain = list(line_list[-2]) + list(line_list[-1])
            chain_info[name] = chain_remain

    # 处理pdb文件，去除多余的链，只保留氨基酸
    pdbfiles = os.listdir(pdb_path)
    for filename in pdbfiles:
        pdbfile = os.path.join(pdb_path, filename)
        pdb_clean = os.path.join(out_path, filename)
        name = filename[:-4]
        chain_remain = chain_info[name]
        extract_aa_chains(pdbfile, pdb_clean, chain_remain)
        num += 1
        print(f'extract chain（{num} / {len(pdbfiles)}） {name}')


aaname = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
          'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
          'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
          'TRP': 'W', 'TYR': 'Y'}


# 定义函数计算两个原子之间的距离
def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)


def extract_info_from_pdb(pdb_file):
    """
    得到序列，序列编号，原子坐标
    """

    res_name = []  # Nseq
    res_num = []  # Nseq
    chain_id = []  # Nseq
    coor = []
    pdbid = os.path.basename(pdb_file)[:-4]
    pdbid_info = {}

    # 创建PDB解析器对象
    parser = PDBParser()

    # 解析PDB文件
    structure = parser.get_structure('STRUCTURE', pdb_file)

    # 获取模型列表中的第一个模型（PDB文件通常只有一个模型）
    model = structure[0]

    # 遍历模型中的所有链
    for chain in model:
        # 获取链中的所有残基
        residues = list(chain.get_residues())

        # 遍历残基，提取序列信息
        chain_sequence = ''
        chain_num = []
        atom_coordinates = []

        for residue in residues:
            # 获取氨基酸的名称
            amino_acid = residue.get_resname()
            if amino_acid in aaname:
                amino_acid_id = aaname[amino_acid]
            else:
                amino_acid_id = 'X'  # 如果是非标准氨基酸，表示为'X'
            # 获取氨基酸的编号
            residue_number = str(residue.get_id()[1])
            residue_number_2 = str(residue.get_id()[2])
            residue_number = ''.join([residue_number, residue_number_2]).strip()
            # 将氨基酸名称和编号添加到序列列表中
            chain_sequence += amino_acid_id
            chain_num.append(residue_number)
            residue_coor = {}
            for atom in residue:
                # 获取原子名称和坐标
                atom_name = atom.get_name()
                atom_coord = atom.get_coord()
                rounded_list = [round(num, 3) for num in atom_coord.tolist()]

                # residue_coor[atom_name] = atom_coord  # 数据是nparray类型

                residue_coor[atom_name] = rounded_list  # 数据是list

            atom_coordinates.append(residue_coor)

        # 将链的序列信息添加到总序列列表中
        chainid = f'{pdbid}_{chain.get_id()}'
        chain_id.append(chainid)
        res_name.append([chainid, chain_sequence])
        res_num.append([chainid, chain_num])
        coor.append([chainid, atom_coordinates])
        pdbid_info[chainid] = {'res_name': chain_sequence, 'res_num': chain_num, 'coor': atom_coordinates}

    return pdbid_info


def process_pdb_files(pdb_clean_path, out_path):
    files = os.listdir(pdb_clean_path)
    info_dict = {}
    num = 0
    for file in files:
        num += 1
        print(f'prepare_pdb {num} / {len(files)}  {file}')
        name = file[:-4]
        path = os.path.join(pdb_clean_path, file)
        info = extract_info_from_pdb(path)
        info_dict[name] = info

    out_info = os.path.join(out_path, 'pdb_info.json')

    with open(out_info, 'w') as jf:
        json.dump(info_dict, jf)


def mp_nerf_torch(a, b, c, l, theta, chi):  # 通过三个坐标（a、b、c）与旋转矩阵，输出处于另一个平面的d的坐标
    """ Custom Natural extension of Reference Frame.
        Inputs:
        * a: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * b: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * c: (batch, 3) or (3,). point(s) of the plane, connected to d
        * theta: (batch,) or (float).  angle(s) between b-c-d
        * chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
        Outputs: d (batch, 3) or (float). the next point in the sequence, linked to c
    """
    if not ((-np.pi <= theta) * (theta <= np.pi)).all().item():
        raise ValueError(f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}")
    # calc vecs
    ba = b - a
    cb = c - b
    # calc rotation matrix. based on plane normals and normalized
    n_plane = torch.cross(ba, cb, dim=-1)
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    rotate = torch.stack([cb, n_plane_, n_plane], dim=-1)
    rotate = rotate / (torch.norm(rotate, dim=-2, keepdim=True) + 1e-5)
    # print (rotate.shape)
    # calc proto point, rotate. add (-1 for sidechainnet convention)
    # https://github.com/jonathanking/sidechainnet/issues/14
    d = torch.stack([-torch.cos(theta),
                     torch.sin(theta) * torch.cos(chi),
                     torch.sin(theta) * torch.sin(chi)], dim=-1).unsqueeze(-1)
    # extend base point, set length
    return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()


def are_receptors_or_ligands(element1, element2, list1, list2):
    return (element1 in list1 and element2 in list1) or (element1 in list2 and element2 in list2)


def build_CB(Ncoor, CAcoor, Ccoor):
    nres = Ncoor.shape[0]
    l = torch.tensor(2.499, dtype=torch.float, device=Ncoor.device).repeat(nres)
    theta = torch.tensor(34.828 / 180.0 * 3.1415926, dtype=torch.float, device=Ncoor.device).repeat(nres)
    chi = torch.tensor(-122.611 / 180.0 * 3.1415926, dtype=torch.float, device=Ncoor.device).repeat(nres)
    cbcoor = mp_nerf_torch(Ncoor, CAcoor, Ccoor, l, theta, chi)
    return cbcoor[0, :]


def rigidFrom3Points_torch(x1, x2, x3):
    """
    通过三个点的坐标计算出了一个旋转矩阵
    """
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2)
    R = torch.stack((e1, e2, e3), axis=1)
    return R


def get_esm_list(data):
    esm_list = []
    num = 0
    for pdbid in data:
        for chain in data[pdbid]:
            seq = data[pdbid][chain]['res_name']
            esm_list.append([chain, seq])
        num += 1

    return esm_list


def process_hdf5(data, rec_ligand_info, out_path):
    fout = h5py.File(out_path, "w")
    length = len(data)
    count = 0
    error_list = []
    for pdbid in data:
        print(f'edge_feature_prepare {count + 1}/{length} ----- {pdbid}')

        resname = ''
        chain_seq = ''
        resnum = []
        chainid = ''
        five_coor = []
        whole_coors = []
        five_distance = []
        rotation = []
        edge_index = []
        receptor = []
        ligand = []
        core_edge_index = []
        chain_id_l = []
        num = 0
        chain_id = 0
        try:
            for chain in data[pdbid]:
                chain_num = 0
                seq = data[pdbid][chain]['res_name']
                coors = data[pdbid][chain]['coor']
                chain_seq += chain[-1]
                for aa in seq:
                    resname += aa
                    resnum.append(num)

                    chain_id_l.append(chain_id)

                    seq_coor = coors[chain_num]
                    whole_coors.append(seq_coor)

                    # N 原子 0
                    n = seq_coor['N']
                    n = torch.tensor(n, dtype=torch.float32)

                    # CA 原子 1
                    ca = seq_coor["CA"]
                    ca = torch.tensor(ca, dtype=torch.float32)

                    # C 原子 2
                    c = seq_coor['C']
                    c = torch.tensor(c, dtype=torch.float32)

                    # O 原子 3
                    o = seq_coor['O']
                    o = torch.tensor(o, dtype=torch.float32)

                    # CB 原子 4
                    if 'CB' not in seq_coor:
                        cb = build_CB(n, ca, c)
                    else:
                        cb = seq_coor['CB']
                        cb = torch.tensor(cb, dtype=torch.float32)

                    # 整合五个原子坐标
                    fivecoor = torch.stack([n, ca, c, o, cb], dim=0)
                    five_coor.append(fivecoor)

                    # 得到每个残基的旋转矩阵（旋转等变
                    aa_rotation = rigidFrom3Points_torch(n, ca, c)
                    rotation.append(aa_rotation)

                    chainid += chain[-1]

                    num += 1
                    chain_num += 1

                chain_id += 1

            for rec_info in rec_ligand_info:
                if rec_info.startswith(pdbid):
                    rec = rec_info.split('_')[-2]
                    lig = rec_info.split('_')[-1]
                    for i in rec:
                        receptor.append(i)
                    for i in lig:
                        ligand.append(i)
                    break

            for i in resnum:
                for j in resnum:
                    coor1 = five_coor[i][4, :]
                    coor2 = five_coor[j][4, :]
                    dist = calculate_distance(coor1, coor2)

                    # cb 链内10埃 链间12埃
                    if dist < 10 and dist != 0:
                        if are_receptors_or_ligands(chainid[i], chainid[j], receptor, ligand):
                            edge_index.append([i, j])
                    if dist < 12 and dist != 0:
                        if not are_receptors_or_ligands(chainid[i], chainid[j], receptor, ligand):
                            core_edge_index.append([i, j])
                            edge_index.append([i, j])

                    # # ca 之间11埃
                    # if dist < 11 and dist != 0:
                    #     edge_index.append([i, j])
                    #     if not are_receptors_or_ligands(chainid[i], chainid[j], receptor, ligand):
                    #         core_edge_index.append([i, j])

            for i in core_edge_index:
                i1 = i[0]
                i2 = i[1]
                coors1 = whole_coors[i1]
                # {"N": [73.227, 31.442, 101.41], "CA": [72.12, 30.453, 101.239], "C": [71.047, 30.634, 102.304], "O": [71.319, 30.52, 103.512]}
                coors2 = whole_coors[i2]

                # 将tensor堆叠成一个nx3的tensor
                tensors1 = [torch.tensor(value) for value in coors1.values()]
                result_tensor1 = torch.stack(tensors1)
                tensors2 = [torch.tensor(value) for value in coors2.values()]
                result_tensor2 = torch.stack(tensors2)

                # 计算成对距离
                edge_dist_feat = torch.cdist(result_tensor1, result_tensor2)

                # 将距离矩阵展平并排序
                sorted_distances, indices = torch.sort(torch.flatten(edge_dist_feat))

                # 获取最小的五个距离
                five_smallest_distances = sorted_distances[:5]

                # 如果需要，可以将这些值转换为Python列表
                five_smallest_distances_list = five_smallest_distances.tolist()
                five_distance.append(five_smallest_distances_list)

            group = fout.create_group(pdbid)

            # 写入数据
            group.create_dataset("resnum", data=np.array(resnum))
            group.create_dataset("chain", data=chain_seq, dtype=h5py.string_dtype(encoding='utf-8'))
            group.create_dataset("five_coor", data=np.array(five_coor))
            group.create_dataset("five_distance", data=np.array(five_distance))
            group.create_dataset("rotation", data=np.array(rotation))
            group.create_dataset("edge_index", data=np.array(edge_index))
            group.create_dataset("core_edge_index", data=np.array(core_edge_index))
            group.create_dataset("batch_index", data=np.array(chain_id_l))

            count += 1

        except Exception as e:
            error_list.append([pdbid, e])
            print(f'{pdbid} 出错啦')

    return error_list


def read_hdf5(hdf5):
    batch = {}
    pdbid = []
    # 打开HDF文件
    with h5py.File(hdf5, 'r') as f:
        num = 0
        for group_name in f:
            num += 1
            pdbid.append(group_name)
            group = f[group_name]

            # 遍历组中的所有数据集
            for dataset_name in group:
                dataset = group[dataset_name]
                if dataset_name not in batch:
                    batch[dataset_name] = []
                if dataset_name == 'chain':
                    batch[dataset_name].append(dataset[()].decode('utf-8'))
                else:
                    batch[dataset_name].append(dataset[()])
    # print(batch)
    return batch, pdbid


def compute_inner_edge_feat(five_atom_coords, edge_idx, core_edge_index, r):
    five_atom_coords = torch.from_numpy(five_atom_coords)
    r = torch.from_numpy(r)
    list_core = core_edge_index.tolist()
    list_whole = edge_idx.tolist()
    list_inner = [i for i in list_whole if i not in list_core]

    inner_edge_idx = torch.tensor(list_inner).to(torch.long).t()
    edge_idx = torch.tensor(list_whole).to(torch.long).t()
    core_edge_index = torch.tensor(list_core).to(torch.long).t()

    src_idx, dst_idx = inner_edge_idx[0], inner_edge_idx[1]

    # Compute distance between each pair of 'true' atoms in neighboring residues.
    five_atom_coords_i, five_atom_coords_j = (
        five_atom_coords[src_idx],
        five_atom_coords[dst_idx],
    )

    edge_dist_feat = torch.cdist(five_atom_coords_i, five_atom_coords_j)

    # Add small Noise
    edge_dist_feat = (edge_dist_feat + torch.randn_like(edge_dist_feat) * 0.02).clip(
        min=0.0
    )

    edge_dist_feat = rbf(edge_dist_feat)  # rbf把每个distance扩了16维
    edge_dist_feat = edge_dist_feat.view(len(edge_dist_feat), -1)  # nedge, 25*16
    # print(f'inner_dis_feature  {edge_dist_feat.size()}')

    edge_angle_feat, edge_dir_fea = cal_angle_direction_feature(inner_edge_idx, five_atom_coords, r)

    info = torch.zeros(len(edge_dir_fea), 1)

    edge_feature = torch.cat([edge_dist_feat, edge_angle_feat, edge_dir_fea, info], dim=1)

    # print(edge_feature.shape)

    return edge_feature, edge_idx, inner_edge_idx, core_edge_index


def rbf(dist, d_min=0, d_max=20, d_count=16):
    d_mu = torch.linspace(d_min, d_max, d_count).reshape(1, 1, 1, -1).to(dist.device)
    d_sigma = (d_max - d_min) / d_count
    dist = dist[:, :, :, None]

    return torch.exp(-((dist - d_mu) ** 2) / (2 * d_sigma ** 2))


def compute_core_edge_feat(five_atom_coords, five_dis, core_edge_idx, r):
    five_atom_coords = torch.from_numpy(five_atom_coords)
    five_dis = torch.from_numpy(five_dis)
    r = torch.from_numpy(r)

    core_edge_index_pt = torch.from_numpy(core_edge_idx)

    # 确保索引是长整数类型
    core_edge_index_pt = core_edge_index_pt.to(torch.long)

    # 转置张量，以便于使用 src_idx, dst_idx = core_edge_idx[0], core_edge_idx[1]
    core_edge_idx = core_edge_index_pt.t()

    src_idx, dst_idx = core_edge_idx[0], core_edge_idx[1]

    # Compute distance between each pair of 'true' atoms in neighboring residues.
    five_atom_coords_i, five_atom_coords_j = (
        five_atom_coords[src_idx],
        five_atom_coords[dst_idx],
    )

    five_dis = five_dis.unsqueeze(1)

    edge_dist_feat = torch.cdist(five_atom_coords_i, five_atom_coords_j)
    edge_dist_feat = torch.cat((edge_dist_feat, five_dis), dim=1)

    # Add small Noise
    edge_dist_feat = (edge_dist_feat + torch.randn_like(edge_dist_feat) * 0.02).clip(
        min=0.0
    )

    edge_dist_feat = rbf(edge_dist_feat)  # rbf把每个distance扩了16维
    edge_dist_feat = edge_dist_feat.view(len(edge_dist_feat), -1)  # nedge, 30*16
    # print(f'core_dis_feature  {edge_dist_feat.size()}')

    edge_angle_feat, edge_dir_fea = cal_angle_direction_feature(core_edge_idx, five_atom_coords, r)

    info = torch.ones(len(edge_dir_fea), 1)

    edge_feature = torch.cat([edge_dist_feat, edge_angle_feat, edge_dir_fea, info], dim=1)

    # torch.set_printoptions(threshold=float('inf'))

    # print(edge_feature.shape)

    return edge_feature


def cal_angle_direction_feature(index, five_coors, r):
    src_idx, dst_idx = index[0], index[1]

    # angle feature
    ri, rj = r[src_idx], r[dst_idx]
    ri_inv = torch.linalg.inv(ri)
    rij = ri_inv @ rj
    edge_angle_feat = rij.view(len(rij), -1)  # nedge, 3*3
    # print(f'angle_feature  {edge_angle_feat.shape}')

    # direction feature
    ca_i = five_coors[src_idx][:, 1:2, :]  # n, 1, 3
    five_j = five_coors[dst_idx]  # n, 5, 3
    vectors = five_j - ca_i  # 计算 ca_i 与 five_j 中每个原子的向量
    dir_feature = vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-6)  # 对每个向量进行归一化，以得到单位向量

    # dir_feature = torch.matmul(ri_inv, dir_feature.permute(0, 2, 1)).permute(0, 2, 1)
    dir_feature = torch.einsum('nij,nmj->nmi', ri_inv, dir_feature)

    edge_dir_fea = dir_feature.reshape(len(dir_feature), -1)  # nedge, 5*3
    # print(f'dir_feature  {edge_dir_fea.shape}')

    return edge_angle_feat, edge_dir_fea


# 使用脚本
txt_file_path = '../input/chain_info.txt'
pdb_files_path = '../input/pdb'
pdb_clean_files_path = '../input/tmp/pdb_clean'
os.makedirs(pdb_clean_files_path, exist_ok=True)
shutil.copy(txt_file_path, '../input/tmp/chain_info.txt')
root_path = '../input/tmp/'
pdb_clean = os.path.join(root_path, 'pdb_clean')
pdb_info = os.path.join(root_path, 'pdb_info.json')
whole_hdf5 = os.path.join(root_path, 'whole.hdf5')
esm_out = os.path.join(root_path, 'seq_list_for_esm.json')
error_path = '../output/error.txt'
edge_feature = '../graph/whole_edge_feature.hdf5'
node_feature = '../graph/esm_feature'
os.makedirs(node_feature, exist_ok=True)

# 调用函数
if not os.path.exists(pdb_info):
    process_dataset(txt_file_path, pdb_files_path, pdb_clean_files_path)
    process_pdb_files(pdb_clean, root_path)


with open(pdb_info, 'r') as file:
    data = json.load(file)


rec_ligand_info = []
with open('../input/tmp/chain_info.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        rec_ligand_info.append(line)


# 用于计算esm特征
esmlist = get_esm_list(data)
with open(esm_out, 'w') as f:
    json.dump(esmlist, f)

# 用于计算边特征
error_result = process_hdf5(data, rec_ligand_info, out_path=whole_hdf5)

# 输出有错的文件
if len(error_result) > 0:
    with open(error_path, 'w') as f:
        for i in error_result[:-1]:
            i = str(i)
            f.write(i + '\n')
        for i in error_result[-1:]:
            i = str(i)
            f.write(i)
    print('find missing atom in pdb file, plz open PPAP/output/error.txt for more info')

assert len(error_result) == 0

batch, pdbid = read_hdf5(whole_hdf5)
fout = h5py.File(edge_feature, "w")
affinity_txt = '../input/for_train.txt'

count = 0
lenth = len(pdbid)
error = []

for i, j, k, m, n, id, b_index, chain_seq in zip(batch['five_coor'], batch['five_distance'], batch['core_edge_index'], batch['edge_index'], batch['rotation'], pdbid, batch['batch_index'], batch['chain']):

    try:
        if os.path.exists(affinity_txt):
            with open(affinity_txt, 'r') as ff:
                lines = ff.readlines()
                for line in lines:
                    line = line.strip()
                    _name = line.split('_')[:-2]
                    _name = '_'.join(_name)
                    _affinity = line.split('_')[-1]
                    if _name == id:
                        affinity = [float(_affinity)]
                        break
        else:
            affinity = [0]

        count += 1
        print(f"saving edge feature （{count} / {lenth}）  {id}")

        core_edge_fea = compute_core_edge_feat(i, j, k, n)
        inner_edge_feature, edge_idx, inner_edge_idx, core_edge_index = compute_inner_edge_feat(i, m, k, n)

        group = fout.create_group(id)

        group.create_dataset("core_edge_fea", data=np.array(core_edge_fea), compression="lzf")
        group.create_dataset("inner_edge_fea", data=np.array(inner_edge_feature), compression="lzf")
        group.create_dataset("edge_index", data=np.array(edge_idx), compression="lzf")
        group.create_dataset("inner_edge_index", data=np.array(inner_edge_idx), compression="lzf")
        group.create_dataset("core_edge_index", data=np.array(core_edge_index), compression="lzf")
        group.create_dataset("affinity", data=np.array(affinity), compression="lzf")
        group.create_dataset("batch_index", data=np.array(b_index), compression="lzf")
        group.create_dataset("chain", data=chain_seq, dtype=h5py.string_dtype(encoding='utf-8'))

    except Exception as e:
        error.append([id, e, type(e)])


if len(error) != 0:
    print(error)
    print('edge_feature_error, plz check your chain info and pdb file')

assert len(error) == 0


"""
node feature
"""

print('loading ESM2-3B')
model_name = "esm2_t36_3B_UR50D"
model_pretrain, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model_pretrain.eval()


def get_embed(datatmp_list, save_path):
    """
    datatmp_list:[[id1, seq1],
                [id2, seq2],
                ...
                [idn, seqn]]  # 一个列表，包含【序列的名称（1a4y_A）, 序列（DILPCVPFSVAKSVKS...LYLGRMFS）】
    """
    len_list = len(datatmp_list)
    num = 0
    for x in datatmp_list:
        batch_labels, batch_strs, batch_tokens = batch_converter([x])
        with torch.no_grad():
            results = model_pretrain(batch_tokens, repr_layers=[36], return_contacts=True)
        token_representations = results["representations"][36]
        for i, (id, seq) in enumerate([x]):
            num += 1
            seq_representation = token_representations[i, 1: len(seq) + 1]  # .mean(0)
            embedding = os.path.join(save_path, f'{id}.pt')
            torch.save(seq_representation, embedding)
            print(f'node_feature {num} / {len_list} {id}')


with open(esm_out, 'r') as f:
    datatmp_list = json.load(f)

get_embed(datatmp_list, node_feature)
