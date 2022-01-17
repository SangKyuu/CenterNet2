from collections import defaultdict

import json
import cv2
import numpy as np
import tqdm
import re
import glob
import image_crop as ut


def plot_images(json_data, paths=None, fname='images.jpg'):
    img_json = {}
    img_path = '../data/phill/images/'
    class_data_path = '../data/phill/class_data/'
    with open(class_data_path+'phill.names','r') as f:
        class_list = f.readlines()

    for data in json_data:
        if data['image_id'] not in img_json:
            img_json[data['image_id']]=[]
        img_json[data['image_id']].append([data['category_id'], *data['bbox'], data['score']])

    for img_num in img_json:
        img = cv2.imread(img_path + str(img_num) + '.png')
        overlay = img.copy()

        bbox_num = len(img_json[img_num])
        # class_num = int(img_num.split('_')[0]) // 1000000
        # real_img_num = img_num % 1000000
        # gt_bbox_num = len(glob.glob(class_data_path
        #                             +class_list[class_num].replace('\n','')+'/annotation/'
        #                             +str(real_img_num)+'_*.xml'))
        for bbox in img_json[img_num]:
            class_num = bbox[0]
            x,y,w,h = map(int,bbox[1:5])   #top left x,y
            c_x, c_y = 0.5*(2*x+w), 0.5*(2*y+h)
            n_x1, n_y1, n_x2, n_y2 = map(int,ut.c_xywh2xyxy(c_x, c_y, w*0.3, h*0.3))
            prob = bbox[-1]
            if int(img_num.split('_')[0]) // 10**6 == class_num:
                color = (200, 0, 0)
            else:
                color = (0, 200, 0)
            # cv2.circle(img, (int(x+0.5*w),int(y+0.5*h)), 3, color, -1)
            # cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
            cv2.rectangle(overlay, (n_x1, n_y1), (n_x2, n_y2), color, -1)  # inside rectangle
            image_new = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

            # cv2.putText(img, str(class_num)+' '+str(prob), (x, y - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)targets[:, 0]
        # cv2.putText(img, str(bbox_num)+'/'+str(gt_bbox_num),(5,20),
        #             cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1)
        cv2.imwrite('./thresh_box/'+str(img_num)+'.png',image_new)
        print('{} Done'.format(img_num))

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


def bind_images(predictions, gt_dict, confidence_th=0.5, rectangle=False, map_gt=True, debug=False):
    img_dict = defaultdict(list)
    wrong_preds = []

    # eliminate low confidence boxes and reconstruct prediction result
    for pred in predictions:
        if pred['score'] > confidence_th:
            img_dict[pred['image_id']].append(pred['bbox'])
        else:
            pass

    # draw boxes
    for img_i in tqdm.tqdm(img_dict):
        dir = gt_dict['images'][img_i]['file_name']
        img = cv2.imread(dir)
        filename = dir.split('/')[-1]
        overlay = img.copy()
        for bbox in img_dict[img_i]:
            if rectangle:
                cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (200, 0, 0), 1)  # inside rectangle
            else:
                cv2.circle(overlay, (int(bbox[0]+0.5*bbox[2]),
                                     int(bbox[1]+0.5*bbox[3])), 5, (0, 200, 0), -1)

            img_new = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        if map_gt:
            with open(re.sub(r'.jpg$|.png$|.JPG$', '.txt', dir), 'r') as f:
                gt_boxes = f.readlines()

            img_new = cv2.putText(img_new, str(len(img_dict[img_i])) + '/' + str(len(gt_boxes)), (5, 20),
                                  cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1)

            if len(img_dict[img_i]) != len(gt_boxes):
                wrong_preds.append(dir+'\n')

        if debug:
            cv2.imshow('DEBUG', img_new)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()

        cv2.imwrite(TEST_RESULT_PATH+filename, img_new)

    return wrong_preds


if __name__ == '__main__':
    BASEPATH = '../../output/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST/inference_dense_test_99/'
    with open(BASEPATH+'coco_instances_results.json', 'r') as f:
        predicted_data = json.load(f)

    with open(BASEPATH+'dense_test_99_coco_format.json', 'r') as f:
        gt_data = json.load(f)

    TEST_RESULT_PATH = BASEPATH + 'test_result/'

    wrongs = bind_images(predicted_data, gt_data)

    with open(TEST_RESULT_PATH+'wrong.txt', 'w') as f:
        f.writelines(wrongs)
