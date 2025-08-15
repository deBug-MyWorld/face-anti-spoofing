import base64
import time
import cv2
import torch
from PIL import Image as Im

from spoofing import FasNet
from ultralytics import YOLO
from utils.resnet50_detect import build_net, pre_process, post_process, compare_embedding, get_scores
from utils.face_util import get_aligned_face

anti_spoof_model = FasNet.Fasnet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_face = YOLO('weights/yolov8n_face.pt')  # 人脸识别
model_face.to(device)
print('YOLO model loaded')
# 加载模型
face_model = build_net("weights/resnet50.pth", [112, 112], 512)
face_model = face_model.to(device)
face_model.eval()
data_transform = pre_process([112, 112])
print('ResNet model loaded')
score_thresh = 0.75
face_size = (112, 112)
spoofing_thresh = 0.35


def run(data):
    return data


def detect_face(frame):
    results = model_face(frame, verbose=False)
    boxes = results[0].boxes.cpu().numpy()
    key_points = results[0].keypoints.xy[0].cpu().numpy()
    return boxes, key_points


def compare_face_from_queue(user_id, q):
    descriptors = []
    faces = []
    face_change = 0
    flag = True  # 标记是否是第一次迭代
    true_spoofing_cnt = []
    res_data = {
        'user_id': user_id,
        'message': None,
        'image': None
    }
    print(q.qsize())
    while not q.empty():
        # 从队列中获取帧
        data = q.get()
        frame = data['frame']
        boxes = data['boxes']
        key_points = data['key_points']

        if len(boxes) == 0:
            print("AlgMain No faces detected")
        # 处理检测到的每一张人脸
        else:
            if len(boxes) > 1:
                print('AlgMain Multiple faces detected')
                # cv2.imwrite("result/" + str(time.time()) + "_" + "multiple.jpg", frame)
            else:
                # 检测是否为假照片
                x, y, w, h = map(int, boxes[0].xywh[0])
                is_real, antispoof_score = anti_spoof_model.analyze(img=frame, facial_area=(x, y, w, h))
                pred_res = 0 if is_real else 1
                true_spoofing_cnt.append(pred_res)
                # if pred_res == 1:
                #     cv2.imwrite("result/fake_test/" + str(time.time()) + ".jpg", crop_img)
                # 人脸对齐，特征检测
                crop_img = get_aligned_face(frame, key_points, face_size)
                face_tensor = data_transform(Im.fromarray(crop_img))
                face_tensor = face_tensor.unsqueeze(0)
                embeddings = face_model(face_tensor.to(device))
                embeddings = post_process(embeddings)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                v = embeddings.detach().cpu().numpy()
                # 将第一张人脸照片直接保存
                if flag:
                    descriptors.append(v)
                    faces.append(frame)
                    flag = False
                else:
                    sign = True  # 用来标记当前人脸是否为新的
                    l = len(descriptors)
                    for i in range(l):
                        dist = compare_embedding(descriptors[i], v)
                        # 将距离映射为人脸相似性分数
                        score = get_scores(dist)
                        if score > score_thresh:
                            face_gray = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                            # 比较两张人脸的清晰度，保存更清晰的人脸
                            if cv2.Laplacian(gray, cv2.CV_64F).var() > cv2.Laplacian(face_gray,
                                                                                     cv2.CV_64F).var():
                                faces[i] = frame
                                descriptors[i] = v
                                face_change += 1
                            sign = False
                            break
                    if sign:
                        print("new face not ok")
                        # descriptors.append(v)
                        # faces.append(frame)
                        res_data['message'] = '认证失败，检测过程只允许同一个人!'
                        return res_data
    print(len(descriptors))  # 输出不同的人脸数
    print(face_change)
    print(true_spoofing_cnt)
    count_ones = sum(true_spoofing_cnt)  # 利用列表中1的真值性，sum()会加总所有为True的元素，即所有1
    # 计算比例
    ratio_ones = count_ones / len(true_spoofing_cnt)
    # 打印结果
    print("1的比例:", ratio_ones)
    if ratio_ones > spoofing_thresh:
        res_data['message'] = '伪造人脸不通过!'
        return res_data
    # 将不同的比较清晰的人脸照片输出到本地
    if len(faces) == 1:
        _, buffer = cv2.imencode('.jpg', faces[0])
        frame_data = base64.b64encode(buffer).decode('utf-8').replace('\n', '').replace('\r', '')
        res_data['image'] = 'data:image/jpeg;base64,' + frame_data
        res_data['message'] = '检测完成'
    # j = 1
    # for fc in faces:
    #     cv2.imwrite("result/" + str(user_id) + "_" + str(j) + ".jpg", fc)
    #     j += 1
    return res_data
