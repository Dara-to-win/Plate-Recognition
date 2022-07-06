from surface import *


def merge(list_plate):
    if list_plate is None:
        return ''
    result = ''
    for e in list_plate:
        result += e
    return result


if __name__ == '__main__':
    img_path = './algorithm_test'
    # 首先遍历当前目录所有文件
    file_list = os.listdir(img_path)
    # 测试集的图片总数
    total = len(file_list)
    # 传统方法检测车牌+SVM识别车牌字符预测正确数
    accuracy_A = 0
    # YOLOv5方法检测车牌+SVM识别车牌字符预测正确数
    accuracy_B = 0
    # YOLOv5方法检测车牌+CNN识别车牌字符预测正确数
    accuracy_C = 0
    win = tk.Tk()
    surface = Surface(win)

    for file in file_list:
        # 利用os.path.join()方法取得路径全名
        cur_path = img_path + '/' + file
        filename = file.split('.')[0]
        result_A, _ = surface.basic_identify_img(cur_path)
        result_B, _ = surface.yolo_svm_identify_img(cur_path)
        result_C, _ = surface.yolo_cnn_identify_img(cur_path)
        result_A = merge(result_A)
        result_B = merge(result_B)
        print("result_A: ", result_A)
        print("result_B: ", result_B)
        print("result_C: ", result_C)
        if result_A == filename:
            accuracy_A += 1
        if result_B == filename:
            accuracy_B += 1
        if result_C == filename:
            accuracy_C += 1

print("A算法的准确率：", accuracy_A / total)
print("B算法的准确率：", accuracy_B / total)
print("C算法的准确率：", accuracy_C / total)
