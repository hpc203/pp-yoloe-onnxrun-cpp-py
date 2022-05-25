import cv2
import numpy as np
import argparse
import onnxruntime

class PP_YOLOE():
    def __init__(self, model_path, label_path, prob_threshold=0.8):
        with open(label_path, 'rt') as f:
            self.class_names = f.read().rstrip('\n').split('\n')
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(model_path, so)
        self.input_size = (640, 640) ###width, height
        self.mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.confThreshold = prob_threshold

    def preprocess(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = img / 255.
        img -= self.mean_[None, None, :]
        img /= self.std_[None, None, :]
        img = np.transpose(img, [2, 0, 1])
        scale_factor = np.array([1., 1.], dtype=np.float32)
        return img, scale_factor

    def detect(self, srcimg):
        img, scale_factor = self.preprocess(srcimg)
        inputs = {'image': img[None, :, :, :], 'scale_factor': scale_factor[None, :]}
        ort_inputs = {i.name: inputs[i.name] for i in self.session.get_inputs() if i.name in inputs}
        output = self.session.run(None, ort_inputs)
        bbox, bbox_num = output
        keep_idx = (bbox[:, 1] > self.confThreshold) & (bbox[:, 0] > -1)
        bbox = bbox[keep_idx, :]
        ratioh = srcimg.shape[0] / self.input_size[1]
        ratiow = srcimg.shape[1] / self.input_size[0]
        for (clsid, score, xmin, ymin, xmax, ymax) in bbox:
            xmin = int(xmin * ratiow)
            ymin = int(ymin * ratioh)
            xmax = int(xmax * ratiow)
            ymax = int(ymax * ratioh)
            cv2.rectangle(srcimg, (xmin,ymin), (xmax,ymax), (0,0,255), thickness=2)
            cv2.putText(srcimg, self.class_names[int(clsid)]+': '+str(round(score,2)), (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), thickness=1)
            print(self.class_names[int(clsid)])
        return srcimg

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='imgs/dog.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='model/ppyoloe_crn_s_300e_coco.onnx', help="onnx filepath")
    parser.add_argument('--classfile', type=str, default='coco.names', help="classname filepath")
    parser.add_argument('--confThreshold', default=0.7, type=float, help='class confidence')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    net = PP_YOLOE(args.modelpath, args.classfile, prob_threshold=args.confThreshold)
    srcimg = net.detect(srcimg)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()