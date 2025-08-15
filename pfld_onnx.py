import time
import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# setup the parameters
resize = transforms.Resize([112, 112])
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
provider = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers()\
    else ['CPUExecutionProvider']
print(provider)
ort_session_landmark = ort.InferenceSession("weights/pfld.onnx", providers=provider)


class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    # landmark of (5L, 2L) from [0,1] to real range
    def reprojectLandmark(self, landmark):
        landmark_ = np.asarray(np.zeros(landmark.shape))
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.x
            y = point[1] * self.h + self.y
            landmark_[i] = (x, y)
        return landmark_


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def landmarks_detect(faces, orig_image):
    for k, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.xyxy[0])
        out_size = 112
        img = orig_image.copy()
        height, width, _ = img.shape
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(max([w, h]) * 1.1)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        cropped_face = cv2.resize(cropped, (out_size, out_size))

        if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
            continue
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        cropped_face = Image.fromarray(cropped_face)
        test_face = resize(cropped_face)
        test_face = to_tensor(test_face)
        test_face.unsqueeze_(0)

        start = time.time()
        ort_inputs = {ort_session_landmark.get_inputs()[0].name: to_numpy(test_face)}
        ort_outs = ort_session_landmark.run(None, ort_inputs)
        end = time.time()
        # print('Time: {:.6f}s.'.format(end - start))
        landmark = ort_outs[0]
        landmark = landmark.reshape(-1, 2)
        landmark = new_bbox.reprojectLandmark(landmark)
        return landmark

