import cv2
import numpy as np
from PIL import Image

#*************************将图片中大于0的像素转为255***************************************************#
mask_path = r"D:\dataset\DIC-C2DH-HeLa1\test\mask\man_seg002.tif"
threshold = 1
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)
mask = Image.open(mask_path).convert("L")
mask = mask.point(table, "1")
mask.show()


#*********************显示16位tif图********************************************#
# def img_show(path: str, window_name: str):
#     uint16_img = cv2.imread(path, -1)
#     for x in range(uint16_img.shape[0]):  # 图片的高
#         for y in range(uint16_img.shape[1]):  # 图片的宽
#             px = uint16_img[x, y]
#             print(px)  # 这样就能得到每个点的bgr值
#     uint16_img -= uint16_img.min()
#     uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
#     uint16_img *= 255
#     new_uint16_img = uint16_img.astype(np.uint8)
#     cv2.imshow(window_name, new_uint16_img)
#     cv2.waitKey(0)
#     # 释放窗口
#     cv2.destroyAllWindows()
#
#
# img_path = r"D:\dataset\DIC-C2DH-HeLa\01_ERR_SEG\mask012.tif"
# img_show(img_path, "unit16_mask")
#******************************************************************#




