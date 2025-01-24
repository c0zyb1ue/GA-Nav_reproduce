from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import torch
import mmcv
import numpy as np
from shutil import copyfile
import os
import cv2
import datetime

def visualize_gt(annotation_path, image_path, palette, save_path):

    original_image = mmcv.imread(image_path)
    gt_mask = mmcv.imread(annotation_path, flag='unchanged')  # uint8 데이터 로드
    
    h, w, _ = original_image.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    # for class_id, color in enumerate(palette):
    #     print(f"클래스 {class_id}: {color}")

    for class_id, color in enumerate(palette):
        color_seg[gt_mask == class_id] = color

    opacity = 0.5

    blended_image = cv2.addWeighted(original_image, 1 - opacity, color_seg, opacity, 0)

    # OpenCV로 저장하기 전에 RGB -> BGR 변환
    blended_image = blended_image[:, :, ::-1]  # RGB to BGR
    mmcv.imwrite(blended_image, save_path)

    print(f"Result saved to: {save_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('-p', default=".", type=str)
    parser.add_argument('-d', action='store_true')
    parser.add_argument('-s', default="./vis.png", type=str)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='rugd_group',
        help='Color palette used for segmentation map')
    args = parser.parse_args()


    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    img = mmcv.imread(args.img)

    result = inference_segmentor(model, img)

    img_base_name = os.path.basename(args.img) 
    img_name, _ = os.path.splitext(img_base_name)
    save_path_pred = os.path.join(args.p, f"../vis/{img_name}_pred.png")
    save_path_gt = os.path.join(args.p, f"../vis/{img_name}_gt.png")
    
    annotation_path = os.path.join(args.p, f"../data/rugd/RUGD_annotations/{img_name.split('_')[0]}/{img_name}_group6.png")

    # 결과 시각화 및 저장
    #show_result_pyplot(model, args.img, result, get_palette(args.palette), save_dir=save_path, display=args.d)
    # palette = get_palette(args.palette)
    # print("팔레트:", palette)
    # for class_id, color in enumerate(palette):
    #     print(f"클래스 {class_id}: {color}")

    model.show_result(args.img, result, palette=get_palette(args.palette), out_file=save_path_pred, show=False)
    visualize_gt(annotation_path, args.img, palette=get_palette(args.palette), save_path=save_path_gt)

    print(f"Result saved to: {save_path_pred}")


if __name__ == '__main__':
    main()
