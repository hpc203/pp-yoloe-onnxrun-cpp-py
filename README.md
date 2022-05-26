# pp-yoloe-onnxrun-cpp-py
使用ONNXRuntime部署PP-YOLOE目标检测，支持PP-YOLOE-s、PP-YOLOE-m、PP-YOLOE-l、PP-YOLOE-x四种结构，包含C++和Python两个版本的程序

起初想使用OpenCV部署的，可是opencv的dnn模块读取onnx文件总是失败，于是只能使用onnxruntime部署了。
由于模型文件比较大，无法直接上传到仓库，因此把模型文件放在百度云盘里，下载
链接: https://pan.baidu.com/s/1wGwkQ2nzPmCZS8aXnZAB2Q  密码: krua

如果你想自己重新生成.onnx文件，可以参考https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe
里的步骤来生成.onnx文件，生成的.onnx文件后，使用opencv的dnn模块读取会失败的，原因分析，可以参考我的博客文章
https://blog.csdn.net/nihate/article/details/112731327
