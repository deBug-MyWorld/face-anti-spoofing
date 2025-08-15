import numpy as np
import torch
import torchvision.transforms as transforms
from .model_resnet import IR_50


def post_process(embeddings, axis=1):
    '''
    特征后处理函数,l2_norm
    :param embeddings:
    :param axis:
    :return:
    '''
    norm = torch.norm(embeddings, 2, axis, True)
    output = torch.div(embeddings, norm)
    return output


def pre_process(input_size):
    '''
    输入图像预处理函数
    :param input_size:
    :return:
    '''
    data_transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return data_transform


def compare_embedding(emb1, emb2):
    '''
    使用欧式距离比较两个人脸特征的差异
    :param emb1:
    :param emb2:
    :return:返回欧式距离(0,+∞),值越小越相似
    '''
    diff = emb1 - emb2
    dist = np.sum(np.power(diff, 2), axis=1)
    return dist


def get_scores(x, meam=1.40, std=0.2):
    '''
    人脸距离到人脸相似分数的映射
    :param x:欧式距离的值
    :param meam:均值,默认meam=1.40
    :param std: 方差,默认std=0.2
    :return: 返回人脸相似分数(0,1),值越大越相似
    '''
    x = -(x - meam) / std
    # sigmoid
    scores = 1.0 / (1.0 + np.exp(-x))
    return scores


def build_net(model_file, input_size, embedding_size):
    mdoel = IR_50(input_size, embedding_size)
    state_dict = torch.load(model_file, map_location="cpu")
    mdoel.load_state_dict(state_dict)
    return mdoel


