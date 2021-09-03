# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import json
from tqdm import tqdm
from ppstructure.table.table_metric import TEDS
from ppstructure.table.predict_table import TableSystem
from ppstructure.utility import init_args
from ppocr.utils.logging import get_logger

logger = get_logger()


def parse_args():
    class config(object):
        def __init__(self):
            pass
    args = config
    # args.table_char_dict_path = 'ppocr/utils/dict/table_structure_dict_TAL.txt'
    args.table_char_dict_path = 'ppocr/utils/dict/table_structure_dict.txt'
    # args.table_char_type = 'ch'
    args.table_char_type = 'en'
    args.table_max_len = 488
    args.enable_mkldnn = True
    args.use_gpu = True
    args.gpu_mem = 4000
    args.use_tensorrt = False
    # args.table_model_dir = '/home/zhaohj/Documents/checkpoint/paddOCR/TAL/table'
    args.table_model_dir = '/home/zhaohj/Documents/checkpoint/paddOCR/inference/table'
    args.det_algorithm = 'DB'
    args.det_limit_side_len = 736
    args.det_db_thresh = 0.5
    args.det_limit_type = 'min'
    args.det_db_box_thresh = 0.5
    args.det_db_unclip_ratio = 2.0
    args.use_dilation = False
    args.det_db_score_mode = 'fast'
    args.benchmark = False
    args.rec_image_shape = "3, 32, 320"
    args.rec_char_type = 'ch'
    args.rec_batch_num = 20
    args.max_text_length = 20
    args.rec_char_dict_path = './ppocr/utils/ppocr_keys_v1.txt'
    args.use_space_char = True
    args.det_model_dir = '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/det'
    args.rec_algorithm = "CRNN"
    args.rec_model_dir =  '/home/zhaohj/Documents/checkpoint/paddOCR/TAL/rec'
    return args

def main(gt_path, img_root, args):
    teds = TEDS(n_jobs=16)

    text_sys = TableSystem(args)
    # jsons_gt = json.load(open(gt_path))  # gt
    lines = open(gt_path).readlines()
    pred_htmls = []
    gt_htmls = []
    for item in tqdm(lines):
        jsons_gt = json.loads(item)
        img_name = jsons_gt['filename']
        img = cv2.imread(os.path.join(img_root, img_name))
        pred_html = text_sys(img)
        pred_htmls.append(pred_html)
        gt_html = jsons_gt['html']['gt']
        gt_htmls.append(gt_html)
    scores = teds.batch_evaluate_html(gt_htmls, pred_htmls)
    print('teds:', sum(scores) / len(scores))


def get_gt_html(gt_structures, gt_contents):
    end_html = []
    td_index = 0
    for tag in gt_structures:
        if '</td>' in tag:
            if gt_contents[td_index] != []:
                end_html.extend(gt_contents[td_index])
            end_html.append(tag)
            td_index += 1
        else:
            end_html.append(tag)
    return ''.join(end_html), end_html


if __name__ == '__main__':
    gt_path = '/home/zhaohj/Documents/dataset/signed_dataset/TableSegmentation/splerge/v2/tabnet/train.json'
    image_dir = '/home/zhaohj/Documents/dataset/signed_dataset/TableSegmentation/splerge/v2/tabnet/src_img'
    main(gt_path, image_dir, parse_args())
