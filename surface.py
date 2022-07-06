import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import tkinter.font as tkFont

import torch

import predict
import cv2
from PIL import Image, ImageTk
import time

from cnn import prediction
from detect import load_model, detect


class Surface(ttk.Frame):
    pic_path = ""
    view_height = 800
    view_width = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    color_transform = {"green": ("绿牌", "#54be88"), "yellow": ("黄牌", "#986201"), "blue": ("蓝牌", "#0476cc"),
                       "white": ("白牌", "#e7ebe6"), "black": ("黑牌", "#3a423b")}

    def funChooseA(self):
        if not self.ChooseA:
            self.ChooseA = True
            self.ChooseB = False
            self.ChooseC = False
            self.v.set(1)
            self.lab['text'] = '已选择“传统方法检测车牌，SVM识别车牌字符”'
            self.identify_img(self.pic_path)
            # print('传统方法检测车牌，SVM识别车牌字符')

    def funChooseB(self):
        if not self.ChooseB:
            self.ChooseA = False
            self.ChooseB = True
            self.ChooseC = False
            self.v.set(2)
            self.lab['text'] = '已选择“YOLOv5方法检测车牌，SVM识别车牌字符”'
            self.identify_img(self.pic_path)
            # print('YOLOv5方法检测车牌，SVM识别车牌字符')

    def funChooseC(self):
        if not self.ChooseC:
            self.ChooseA = False
            self.ChooseB = False
            self.ChooseC = True
            self.v.set(3)
            self.lab['text'] = '已选择“YOLOv5方法检测车牌，CNN识别车牌字符”'
            self.identify_img(self.pic_path)
            # print('YOLOv5方法检测车牌，CNN识别车牌字符')

    def __init__(self, win):

        self.ChooseA = False
        self.ChooseB = False
        self.ChooseC = False
        self.pic_path = ''

        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("车牌识别")
        # 窗口最大化
        # win.state("zoomed")
        # 设置窗口大小
        win.geometry('1100x800+200+100')
        fontStyle = tkFont.Font(family="Lucida Grande", size=16, weight='bold')
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)
        ttk.Label(frame_left, text='原图：', font=fontStyle).pack(anchor="nw")
        ttk.Label(frame_right1, text='车牌位置：', font=fontStyle).grid(column=0, row=0, sticky=tk.W)

        from_pic_ctl = ttk.Button(frame_right2, text="选择图片", width=20, command=self.from_pic)

        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")
        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='识别结果：', font=fontStyle).grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="", font=('楷体', 16, 'bold'), foreground='red')
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)

        ttk.Label(frame_right1, text='选择识别算法：', font=fontStyle).grid(column=0, row=4, sticky=tk.W)
        self.v = IntVar()
        self.radioBtnA = Radiobutton(frame_right1, text="传统方法检测车牌，SVM识别车牌字符(默认)", variable=self.v, value=1,
                                     command=self.funChooseA, font=('楷体', 14))
        self.radioBtnA.grid(column=0, row=5, sticky=tk.W)
        self.radioBtnB = Radiobutton(frame_right1, text="YOLOv5方法检测车牌，SVM识别车牌字符", variable=self.v, value=2,
                                     command=self.funChooseB, font=('楷体', 14))
        self.radioBtnB.grid(column=0, row=6, sticky=tk.W)
        self.radioBtnC = Radiobutton(frame_right1, text="YOLOv5方法检测车牌，CNN识别车牌字符", variable=self.v, value=3,
                                     command=self.funChooseC, font=('楷体', 14))
        self.radioBtnC.grid(column=0, row=7, sticky=tk.W)
        self.lab = Label(frame_right1, text="", font=('楷体', 14))
        self.lab.grid(column=0, row=8, sticky=tk.W)

        self.radios = [self.radioBtnA, self.radioBtnB, self.radioBtnC]
        # 禁用选择模型的单选框
        for item in self.radios:
            item.config(state='disabled')
        self.radioBtnA.select()
        self.lab['text'] = '已选择“传统方法检测车牌，SVM识别车牌字符”'
        from_pic_ctl.pack(anchor="se", pady="5")

        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

    # 获取图片，调整图片比例，适应窗口大小
    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        width = imgtk.width()
        height = imgtk.height()
        if width > self.view_width or height > self.view_height:
            # 宽度缩小比例
            width_factor = self.view_width / width
            # 高度缩小比例
            height_factor = self.view_height / height
            factor = min(width_factor, height_factor)
            width = int(width * factor)
            if width <= 0:
                width = 1
            height = int(height * factor)
            if height <= 0:
                height = 1
            im = im.resize((width, height), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    # 展示结果，渲染到GUI界面
    def show_roi(self, r, roi):
        if r:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            # 渲染车牌图片
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            result = r[0] + r[1] + '·'
            for i in range(2, len(r)):
                result += r[i]
            self.r_ctl.configure(text=str(result))
            self.update_time = time.time()
        elif self.update_time + 8 < time.time():
            # 超时异常处理
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")

    # 传统算法识别车牌
    def basic_identify_img(self, img_path):
        resize_rates = (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
        r = None
        roi = None
        # 调整比例进行车牌检测
        for resize_rate in resize_rates:
            try:
                # 读取图片
                img_bgr = predict.imreadex(img_path)
                # 识别到的字符、定位的车牌图像、车牌颜色
                r, roi, _ = self.predictor.predict(img_bgr, resize_rate)
            except:
                continue
            if r:
                break
        return r, roi

    # 生成图片输出路径
    # def get_output_path(self, img_path):
    #     output = img_path.split('/')
    #     temp = output[-1]
    #     # 获取图片名称
    #     filename = temp.split('.')
    #     filename = filename[0] + '_warp.' + filename[-1]
    #     output[-1] = filename
    #     output_path = ''
    #     for i in range(0, len(output)):
    #         if i != len(output) - 1:
    #             output_path += output[i] + '/'
    #         else:
    #             output_path += output[i]
    #     return output_path
    def get_output_path(self, img_path):
        output = img_path.split('/')
        temp = output[-1]
        # 获取图片名称
        filename = temp.split('.')
        filename = filename[0] + '_warp.' + filename[-1]
        base_path = os.getcwd()
        output_path = os.path.join(base_path, 'temp', filename)
        return output_path

    # yolo + svm的车牌检测方案
    def yolo_svm_identify_img(self, img_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = './weights/last.pt'
        cfg_path = './models/yolov5s.yaml'
        model = load_model(weights, cfg_path, device)
        output_path = self.get_output_path(img_path)
        # print("img_path: ", img_path)
        # print("output_path: ", output_path)
        # 检测车牌输出到output_path
        detect(model, img_path, device, output_path)
        # 读取路径的图片
        img_bgr = predict.imreadex(output_path)
        colors = ['green', 'blue', 'yellow', 'black', 'white']
        for color in colors:
            # 车牌字符识别
            predict_result, roi, card_color, _ = self.predictor.predict_plate_value(color, img_bgr)
            if len(predict_result) != 0:
                break
        return predict_result, roi

    # yolo + cnn的车牌检测方案
    def yolo_cnn_identify_img(self, img_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = './weights/last.pt'
        cfg_path = './models/yolov5s.yaml'
        model = load_model(weights, cfg_path, device)
        output_path = self.get_output_path(img_path)
        # print("img_path: ", img_path)
        # print("output_path: ", output_path)
        # 检测车牌输出到output_path
        detect(model, img_path, device, output_path)
        # 车牌字符识别
        predict_result, roi, _ = prediction.predict(output_path)
        roi = predict.imreadex(output_path)
        return predict_result, roi

    # 车牌检测算法的控制函数
    def identify_img(self, img_path):
        # 根据单选框的值进行对应的处理
        if self.v.get() == 1:
            # 使用传统的图像处理得到的结果
            r, roi = self.basic_identify_img(img_path)
        elif self.v.get() == 2:
            # 使用yolov5 + svm 进行车牌检测
            r, roi = self.yolo_svm_identify_img(img_path)
        else:
            # 使用yolov5 + cnn 进行车牌检测
            r, roi = self.yolo_cnn_identify_img(img_path)
        self.show_roi(r, roi)

    # 点击选择图片后的处理函数
    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg"), ("png图片", "*.png")])
        # print(self.pic_path)
        if self.pic_path:
            # 读取路径的图片
            img_bgr = predict.imreadex(self.pic_path)
            # 修改图片的比例适应tkinter的窗口大小
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            for item in self.radios:
                item.config(state='normal')
            self.identify_img(self.pic_path)


# 窗口关闭后的处理函数
def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        # 程序执行2s后结束
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()
    surface = Surface(win)
    # 窗口关闭
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()
