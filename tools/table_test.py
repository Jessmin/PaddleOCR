import os
import cv2
from paddleocr import PPStructure, draw_structure_result, save_structure_res

table_engine = PPStructure(show_log=True)

save_folder = './output/table'
img_path = '/home/zhaohj/Documents/dataset/Table/TAL/val_img/val_img/1a1237326b29fdc317f1cefc376f6156.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
# save_structure_res(result, save_folder,
#                    os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = 'doc/fonts/simfang.ttf'  # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result, font_path=font_path)
im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')
im_show.show()
