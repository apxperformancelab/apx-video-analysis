"""Minimal YOLO11 detect / pose inference with onnxruntime (no torch, no ultralytics)."""
import cv2, numpy as np, onnxruntime as ort


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


class OnnxYolo:
    def __init__(self, path, threads=0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.inp = self.s.get_inputs()[0].name
        self.size = self.s.get_inputs()[0].shape[2]

    def _run(self, img):
        lb, r, dx, dy = letterbox(img, self.size)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.s.run(None, {self.inp: np.ascontiguousarray(x)})[0][0].T  # (N, C)
        return out, r, dx, dy

    @staticmethod
    def _nms(boxes, scores, iou=0.5):
        if not len(boxes):
            return []
        xywh = np.c_[boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou)
        return np.array(keep).reshape(-1)

    def persons(self, img, conf=0.15):
        """Detection model (COCO, 80 classes): returns person boxes xyxy (n,4) and scores (n,)."""
        out, r, dx, dy = self._run(img)
        sc = out[:, 4]  # class 0 = person
        m = sc > conf
        b, sc = out[m, :4], sc[m]
        xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
        xyxy = (xyxy - [dx, dy, dx, dy]) / r
        k = self._nms(xyxy, sc)
        return xyxy[k], sc[k]

    def pose(self, img, conf=0.1):
        """Pose model: returns boxes xyxy (n,4), scores (n,), keypoints (n,17,3) [x, y, conf]."""
        out, r, dx, dy = self._run(img)
        sc = out[:, 4]
        m = sc > conf
        o = out[m]; sc = sc[m]
        if not len(o):
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
        b = o[:, :4]
        xyxy = np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]
        xyxy = (xyxy - [dx, dy, dx, dy]) / r
        kp = o[:, 5:].reshape(-1, 17, 3).copy()
        kp[:, :, 0] = (kp[:, :, 0] - dx) / r
        kp[:, :, 1] = (kp[:, :, 1] - dy) / r
        k = self._nms(xyxy, sc)
        return xyxy[k], sc[k], kp[k]
