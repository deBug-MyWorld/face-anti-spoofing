import cv2
import numpy as np
import skimage
import torch
from scipy.spatial import distance as dist

from ultralytics import YOLO


def EAR(eye):
    # 计算眼睛的两组垂直关键点之间的欧式距离
    A = dist.euclidean(eye[1], eye[5])  # 1,5是一组垂直关键点
    B = dist.euclidean(eye[2], eye[4])  # 2,4是一组
    # 计算眼睛的一组水平关键点之间的欧式距离
    C = dist.euclidean(eye[0], eye[3])  # 0,3是一组水平关键点

    return (A + B) / (2.0 * C)


def MAR(mouth):
    # 默认二范数：求特征值，然后求最大特征值得算术平方根
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59（人脸68个关键点）
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55

    return (A + B) / (2.0 * C)


def nose_jaw_distance(nose, jaw):
    # 计算鼻子上一点"27"到左右脸边界的欧式距离
    face_left1 = dist.euclidean(nose[0], jaw[0])  # 27, 0
    face_right1 = dist.euclidean(nose[0], jaw[16])  # 27, 16
    # 计算鼻子上一点"30"到左右脸边界的欧式距离
    face_left2 = dist.euclidean(nose[3], jaw[2])  # 30, 2
    face_right2 = dist.euclidean(nose[3], jaw[14])  # 30, 14
    # 创建元组，用以保存4个欧式距离值
    face_distance = (face_left1, face_right1, face_left2, face_right2)

    return face_distance


class face_quality_assessment():
    def __init__(self, path):
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 112
        self.input_width = 112

    def detect(self, srcimg):
        input_img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        input_img = (input_img.astype(np.float32) / 255.0 - 0.5) / 0.5

        blob = cv2.dnn.blobFromImage(input_img.astype(np.float32))
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outputs[0].reshape(-1)


def get_reference_facial_points(square=True):
    """
    获得人脸参考关键点,目前支持两种输入的参考关键点,即[96, 112]和[112, 112]
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    ==================
    face_size_ref = [112, 112]
    kpts_ref = [[38.29459953 51.69630051]
                [73.53179932 51.50139999]
                [56.02519989 71.73660278]
                [41.54930115 92.3655014 ]
                [70.72990036 92.20410156]]

    ==================
    square = True, crop_size = (112, 112)
    square = False,crop_size = (96, 112),
    :param square: True is [112, 112] or False is [96, 112]
    :param vis: True or False,是否显示
    :return:
    """
    # face size[96_112] reference facial points
    face_size_ref = [96, 112]
    kpts_ref = [[30.29459953, 51.69630051],
                [65.53179932, 51.50139999],
                [48.02519989, 71.73660278],
                [33.54930115, 92.3655014],
                [62.72990036, 92.20410156]]
    kpts_ref = np.asarray(kpts_ref)  # kpts_ref_96_112
    # for output_size=[112, 112]
    if square:
        face_size_ref = np.array(face_size_ref)
        size_diff = max(face_size_ref) - face_size_ref
        kpts_ref += size_diff / 2
        face_size_ref += size_diff

    return np.float32(kpts_ref)


def get_aligned_face(image, keypoint, align_size):
    '''
    Args:
        image: origin imsge
        keypoint: five keypoints with shape of (5, 2)
        align_size: output size of (w, h), exp: (112, 112)
    Returns:
        aligned face with the size of align_size
    '''
    st_kp = get_reference_facial_points()
    st = skimage.transform.SimilarityTransform()  # define the  function of affine transformation
    st.estimate(keypoint, st_kp)  # calculate the matrix of affine transformation
    align_face = cv2.warpAffine(image, st.params[0:2, :], align_size)  # face align
    return align_face
