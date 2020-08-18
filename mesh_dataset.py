import torch
import torch.utils.data
import numpy as np
import os
import glob


class PcdDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.is_val = cfg['is_val']
        self.root_dir = cfg['data_locate']
        if self.is_val:
            path = "val"
        else:
            path = "train"
        self.file_names = glob.glob(self.root_dir + path + "/*/*.pcd")
        self.measure_cnt = cfg['measure_cnt']

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        pcd_path = self.file_names[index]
        (filepath, tempfilename) = os.path.split(pcd_path)
        pcd_meta = dict(img_id=os.path.split(filepath)[1], car_id=os.path.splitext(tempfilename)[0])

        pcd_origin = load_pcd_to_ndarray(pcd_path)
        pcd = np.zeros((self.measure_cnt, 3))
        index = min(self.measure_cnt, pcd_origin.shape[0])

        mean_pcd, max_pcd = np.zeros((self.measure_cnt, 3)), np.ones((self.measure_cnt, 3))
        mean_pcd[:index, :] = np.mean(pcd_origin[:index], axis=0, keepdims=True)
        # min_pcd[:index, :] = np.min(pcd_origin[:index], axis=0, keepdims=True)
        max_pcd[:index, :] = np.max(np.abs((pcd_origin[:index] - mean_pcd[:index])), axis=0, keepdims=True)
        max_tmp = np.max(max_pcd)
        pcd[:index, :] = (pcd_origin[:index] - mean_pcd[:index]) / max_tmp
        mean_ref = torch.from_numpy(mean_pcd[0]).float().view(1, 3)
        if not self.is_val:
            return torch.from_numpy(pcd).float(), mean_ref  # N * 3
        else:
            return torch.from_numpy(pcd).float(), mean_ref, pcd_meta


def read_pcd(pcd_path):
    lines = []
    num_points = None

    with open(pcd_path, 'r') as f:
        for line in f:
            lines.append(line.strip())
            if line.startswith('POINTS'):
                num_points = int(line.split()[-1])
    assert num_points is not None

    points = []
    for line in lines[-num_points:]:
        x, y, z = list(map(float, line.split()))
        points.append(np.array([x, y, z]))
    return np.array(points)  # N * 3


def load_pcd_to_ndarray(pcd_path):
    with open(pcd_path) as f:
        while True:
            ln = f.readline().strip()
            if ln.startswith('DATA'):
                break

        points = np.loadtxt(f)
        points = points[:, 0:3]
        return points


def write_pcd(points, save_pcd_path):
    HEADER = '''\
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z
    SIZE 4 4 4 
    TYPE F F F 
    COUNT 1 1 1 
    WIDTH {}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {}
    DATA ascii
    '''
    n = len(points)
    lines = []
    for i in range(n):
        x, y, z = points[i]
        lines.append('{:.6f} {:.6f} {:.6f}'.format(x, y, z))
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(n, n))
        f.write('\n'.join(lines))


def write_pcd_from_ndarray(points, save_pcd_path):
    HEADER = '''\
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z
    SIZE 4 4 4 
    TYPE F F F 
    COUNT 1 1 1 
    WIDTH {}
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS {}
    DATA ascii
    '''
    with open(save_pcd_path, 'w') as f:
        f.write(HEADER.format(len(points), len(points)) + '\n')
        np.savetxt(f, points, delimiter=' ', fmt='%f %f %f')


def write_ndarray(array, save_path):
    with open(save_path, 'w') as f:
        np.savetxt(f, array, delimiter=' ', fmt='%f %f %f')
