# Plate-Recognition
中国车牌识别

-- algorithm_test文件夹中是测试识别算法的测试集

-- cnn文件夹中是cnn字符识别算法，其中train_dir中存放的预训练好的参数文件13_0.213.hdf5，由于该文件较大，我存放于百度网盘中，链接: https://pan.baidu.com/s/1XlSbPg5nvS7ayyYqRR0GJw?pwd=k9px 提取码: k9px 
      
      license_plate_model.py是检测模型，prediction.py是识别车牌字符的代码。
      
-- models文件夹中存放的是YOLOv5算法相关的模型代码

-- temp文件夹中存放的是识别过程中产生的车牌图片

-- test文件夹中存放的是一些汽车图片，用于测试程序功能

-- train文件夹中存放的是SVM训练用的数据集

-- utils文件夹中存放的是YOLOv5算法依赖的一些工具函数

-- weights文件夹中存放的是YOLOv5算法的预训练好的参数文件last.pt，存放于百度网盘中，链接如上

-- algorithm_test.py是测试车牌检测准确率的代码

-- config.js是设置GUI界面的相关配置

-- detect.py是YOLOv5检测车牌的代码

-- predict.py是传统图像处理检测车牌的代码，里面包含车牌检测和SVM车牌识别的代码

-- requirements.txt是项目所需要的函数库

-- surface.py是GUI界面的代码，运行该文件即可运行程序，相当于整个项目的主函数

-- svm.dat和svmchinese.dat是SVM算法训练好后的参数

更多信息可以看我的博客介绍：https://blog.csdn.net/swust512017/article/details/125637044

参考文献：

[1]基于yolov5的车牌检测，https://github.com/xialuxi/yolov5-car-plate

[2]CCPD（中国城市停车数据集)，https://github.com/detectRecog/CCPD

[3]端到端车牌识别项目，https://github.com/MrZhousf/license_plate_recognize

[4]车牌号识别python + opencv，https://blog.csdn.net/wzh191920/article/details/79589506

[5]车牌号识别https://github.com/wzh191920/License-Plate-Recognition
