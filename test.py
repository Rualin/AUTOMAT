import os
import shutil

import torch
from thop import profile, clever_format
from ultralytics import YOLO


# model = YOLO("my_yolo.yaml")
# model.export(format="onnx")


if __name__ == "__main__":  
    input_shape = 640
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = YOLO("my_yolo.yaml").model.to(device)
    for m_sub in m.modules():
        for attr in ['total_ops', 'total_params']:
            if hasattr(m_sub, attr):
                delattr(m_sub, attr)

    dummy_input = torch.randn((2, 3, input_shape, input_shape)).to(device)  
    flops, params = profile(m.to(device), inputs=(dummy_input,))  
    flops, params = clever_format([flops, params], "%.3f")  
    print('Total GFLOPS: %s' % (flops))  
    print('Total params: %s' % (params))  

# op = os.path

# dir = "images"
# train = "train"
# val = "val"
# imgdir = op.join(train, "images")
# labdir = op.join(train, "labels")
# imgs = sorted(os.listdir(imgdir))
# labs = sorted(os.listdir(labdir))

# valimg = op.join(val, "images")
# vallab = op.join(val, "labels")

# # if op.exists(valdir):
# #     shutil.rmtree(valdir)
# os.makedirs(valimg, exist_ok=True)
# os.makedirs(vallab, exist_ok=True)

# # for i in range(0, len(imgs), 1):
# #     shutil.move(op.join(maindir, imgs[i]), valdir)

# # print(len(os.listdir(maindir)))
# # print(len(os.listdir(valdir)))
# # ind = 0
# for img in imgs:
#     # if ind % 10 == 0:
#     #     ind += 1
#     #     continue
#     # ind += 1
#     name = img[:-4] + ".txt"
#     ind = labs.index(name)
#     # print(ind)
#     # shutil.move(op.join(imgdir, img), valimg)
#     # shutil.move(op.join(labdir, name), vallab)

# print(len(os.listdir(imgdir)))
# print(len(os.listdir(labdir)))
# print(len(os.listdir(valimg)))
# print(len(os.listdir(vallab)))
