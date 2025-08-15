import argparse
import os
import queue
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi_socketio import SocketManager
import cv2
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

import AlgMain
from AlgMain import compare_face_from_queue, detect_face
import numpy as np
import base64
from imutils import face_utils
from utils.face_util import MAR, nose_jaw_distance
from pfld_onnx import landmarks_detect

eye_cascPath = 'weights/haarcascade_eye_tree_eyeglasses.xml'
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
# 嘴的索引
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# 摇头索引
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']


function_constant = {
    "eye_detect_from_camera": {"eye_close": 1, "detect_thresh": 1},
    "mouth_detect_from_camera": {"mouth_thresh": 0.45, "mouth_close": 1, "detect_thresh": 1},
    "head_detect_from_camera": {"detect_thresh": 1}
}
face_dis_far_thresh = 0.52  # 提示离镜头远的阈值
face_dis_close_thresh = 0.43  # 提示离镜头近的阈值
time_frame = 3  # 抽帧频率
TIMEOUT = 30  # 超时时间
user_cnt = {}  # 用户动作计数
user_image = {}  # 保存每个用户的图片

app = FastAPI()
webapp = FastAPI()
contextPath = '/web'
app.mount(contextPath + "/static", StaticFiles(directory="./templates/static"), name="static")
app.mount(contextPath, webapp)
templates = Jinja2Templates(directory="templates")
sio = SocketManager(app=app, mount_location="/")



def eye_detect_from_camera(user_id, shape, frame):
    if (user_cnt[user_id]['eye_detect_from_camera']['total'] <
            function_constant['eye_detect_from_camera']['detect_thresh']):
        (x, y, w, h) = cv2.boundingRect(shape)
        face_frame = frame[y:y + h, x:x + w]
        eyes = eyeCascade.detectMultiScale(face_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(eyes) == 2:  # 检测到睁眼
            user_cnt[user_id]['eye_detect_from_camera']['count_eye'] += 1
        elif len(eyes) == 0:  # 检测到闭眼
            if (user_cnt[user_id]['eye_detect_from_camera']['count_eye'] >=
                    function_constant['eye_detect_from_camera']['eye_close']):
                user_cnt[user_id]['eye_detect_from_camera']['total'] += 1
            user_cnt[user_id]['eye_detect_from_camera']['count_eye'] = 0
        res_data = {
            'eye': user_cnt[user_id]['eye_detect_from_camera']['total'],
            'msg': '请眨眨眼'
        }
        return res_data


def mouth_detect_from_camera(user_id, shape):
    if (user_cnt[user_id]['mouth_detect_from_camera']['total'] <
            function_constant['mouth_detect_from_camera']['detect_thresh']):
        # 提取嘴唇坐标，然后使用该坐标计算嘴唇纵横比
        Mouth = shape[mStart:mEnd]
        mar = MAR(Mouth)
        print('mar', mar)
        # 判断嘴唇纵横比是否高于张嘴阈值，如果是，则增加张嘴帧计数器
        if mar > function_constant['mouth_detect_from_camera']['mouth_thresh']:
            user_cnt[user_id]['mouth_detect_from_camera']['count_mouth'] += 1

        else:
            # 如果张嘴帧计数器不等于0，则增加张嘴的总次数
            if (user_cnt[user_id]['mouth_detect_from_camera']['count_mouth'] >=
                    function_constant['mouth_detect_from_camera']['mouth_close']):
                user_cnt[user_id]['mouth_detect_from_camera']['total'] += 1
            user_cnt[user_id]['mouth_detect_from_camera']['count_mouth'] = 0
        res_data = {
            'mouth': user_cnt[user_id]['mouth_detect_from_camera']['total'],
            'msg': '请张张嘴'
        }
        return res_data


def head_detect_from_camera(user_id, shape):
    if (user_cnt[user_id]['head_detect_from_camera']['total'] <
            function_constant['head_detect_from_camera']['detect_thresh']):
        # 提取鼻子和下巴的坐标，然后使用该坐标计算鼻子到左右脸边界的欧式距离
        nose = shape[nStart:nEnd]
        jaw = shape[jStart:jEnd]
        NOSE_JAW_Distance = nose_jaw_distance(nose, jaw)
        # 移植鼻子到左右脸边界的欧式距离
        face_left1 = NOSE_JAW_Distance[0]
        face_right1 = NOSE_JAW_Distance[1]
        face_left2 = NOSE_JAW_Distance[2]
        face_right2 = NOSE_JAW_Distance[3]

        # 根据鼻子到左右脸边界的欧式距离，判断是否摇头
        # 左脸大于右脸
        if face_left1 >= face_right1 + 2 and face_left2 >= face_right2 + 2:
            user_cnt[user_id]['head_detect_from_camera']['distance_left'] += 1
            # 右脸大于左脸
        if face_right1 >= face_left1 + 2 and face_right2 >= face_left2 + 2:
            user_cnt[user_id]['head_detect_from_camera']['distance_right'] += 1
            # 左脸大于右脸，并且右脸大于左脸，判定摇头
        if (user_cnt[user_id]['head_detect_from_camera']['distance_left'] != 0 and
                user_cnt[user_id]['head_detect_from_camera']['distance_right'] != 0):
            user_cnt[user_id]['head_detect_from_camera']['total'] += 1
            user_cnt[user_id]['head_detect_from_camera']['distance_left'] = 0
            user_cnt[user_id]['head_detect_from_camera']['distance_right'] = 0
        res_data = {
            'head': user_cnt[user_id]['head_detect_from_camera']['total'],
            'msg': '请摇摇头'
        }
        return res_data


def action_controller(selected_functions, user_id, shape, frame):
    res = None
    for func in selected_functions:
        if (function_constant[function_map[func].__name__]['detect_thresh'] >
                user_cnt[user_id][function_map[func].__name__]['total']):
            if func == 'eye':
                res = eye_detect_from_camera(user_id, shape, frame)
            else:
                res = function_map[func](user_id, shape)
            break
        else:
            continue
    return res


function_map = {
    "eye": eye_detect_from_camera,
    "mouth": mouth_detect_from_camera,
    "head": head_detect_from_camera
}


def result_map(user_id, selected_functions, message, msg_type):
    return {
        'user_id': user_id,
        'selected_functions': selected_functions,
        'message': message,
        'type': msg_type}


def del_map(user_id):
    try:
        del user_cnt[user_id]
        del user_image[user_id]
    except KeyError:
        print("键不存在")


# 当客户端连接时，初始化活动时间
@sio.on('connect', namespace="/")
async def on_connect(sid, *args):
    session_id = sid
    print(f"客户端 {session_id} 已连接")
    global user_cnt, user_image
    functions = list(function_map.keys())
    # selected_functions = random.sample(functions, random.randint(1, 3))
    connect_times = time.time()
    selected_functions = functions
    function_cnt = {
        "eye_detect_from_camera": {"count_eye": 0, "total": 0},
        "mouth_detect_from_camera": {"count_mouth": 0, "total": 0},
        "head_detect_from_camera": {"distance_left": 0, "distance_right": 0, "total": 0},
        "connect_times": connect_times,
        "is_time_out": False
    }
    user_cnt[session_id] = function_cnt
    user_image[session_id] = {
        'sum_frame': 0,
        'queue': queue.Queue()
    }
    res_data = {
        'user_id': session_id,
        'selected_functions': selected_functions
    }
    await sio.emit('connected', res_data, to=sid)


@sio.on('disconnect', namespace="/")
def on_disconnect(sid):
    session_id = sid
    print(f"客户端 {session_id} 已断开连接")
    del_map(session_id)


@sio.on('image', namespace="/")
async def handle_image(sid, data):
    current_time = time.time()
    user_id = data['user_id']
    selected_functions = data['selected_functions']
    image_data = base64.b64decode(data['image'].split(',')[1])
    activity_time = user_cnt[user_id]['connect_times']
    if current_time - activity_time <= TIMEOUT:
        # Decode the image from base64
        nparr = np.frombuffer(image_data, np.uint8)
        if len(nparr) == 0:  # 页面摄像头画面是否出来
            boxes = []
            key_points = None
            frame = None
        else:
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            boxes, key_points = detect_face(frame)
        if len(boxes) == 0:
            print("No face detected.")
            await sio.emit('response', result_map(user_id, selected_functions, '没有检测到人脸', 0), to=sid)
        if len(boxes) > 1:
            print("Multiple face detected.")
            await sio.emit('response', result_map(user_id, selected_functions, '检测到多张人脸', 0), to=sid)
        if len(boxes) == 1:
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            w = x2 - x1
            face_ratio = w / frame.shape[1]
            # 根据比例给出提示
            # print('face_ratio', face_ratio)
            if face_ratio > face_dis_far_thresh:
                print("请离镜头远一些")
                await sio.emit('response', result_map(user_id, selected_functions, '请离镜头远一些', 0), to=sid)
            elif face_ratio < face_dis_close_thresh:
                print("请离镜头近一些")
                await sio.emit('response', result_map(user_id, selected_functions, '请离镜头近一些', 0), to=sid)
            else:
                if user_id in user_cnt and not all(user_cnt[user_id][function_map[func].__name__]['total'] >=
                                                   function_constant[function_map[func].__name__]['detect_thresh']
                                                   for func in selected_functions):
                    user_image[user_id]['sum_frame'] += 1
                    if user_image[user_id]['sum_frame'] % time_frame == 0:
                        user_image[user_id]['queue'].put({'frame': frame, 'key_points': key_points, 'boxes': boxes})
                    landmark = landmarks_detect(boxes, frame).astype(int)

                    res = action_controller(selected_functions, user_id, landmark, frame)

                    # Emit the processed image back to the client
                    await sio.emit('response', result_map(user_id, selected_functions, res, 1), to=sid)
                else:
                    print("detected over")
                    await sio.emit('response', result_map(user_id, selected_functions, None, 0), to=sid)
    else:
        print(f"客户端 {user_id} 超过 {TIMEOUT} 秒，断开连接")
        user_cnt[user_id]['is_time_out'] = True
        await sio.emit('response', result_map(user_id, selected_functions, '动作判定超时！', 0), to=sid)
        await sio.disconnect(user_id)


@sio.on('disconnectUser', namespace="/")
async def handle_disconnect(sid, data):
    user_id = data['user_id']
    print(user_id + '客户端即将断开连接')
    print('sum_frame', user_image[user_id]['sum_frame'])
    if not user_cnt[user_id]['is_time_out']:
        # 调用处理函数
        st = time.time()
        res_data = compare_face_from_queue(user_id, user_image[user_id]['queue'])
        print(time.time() - st)
        # 删除相关字典信息
        await sio.emit('beforeDisconnect', res_data, to=sid)
        del_map(user_id)
    else:
        del_map(user_id)


@webapp.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})




if __name__ == '__main__':
    # uvicorn.run(app='AppStart:app', host="0.0.0.0", port=port)
    uvicorn.run(app='AppStart:app', host="0.0.0.0", port=8000, ssl_keyfile='cert/server.key'
                , ssl_certfile='cert/server.crt')
    # python AppStart.py
