import json
import numpy as np
import os

class hoiw():
    def __init__(self, annotation_file):
        self.annotations = json.load(open(annotation_file, 'r'))
        self.overlap_iou = 0.5
        self.verb_name_dict = {1: 'smoke', 2: 'call', 3: 'play(cellphone)', 4: 'eat', 5: 'drink',
                            6: 'ride', 7: 'hold', 8: 'kick', 9: 'read', 10: 'play (computer)'}
        self.file_name = []
        for gt_i in self.annotations:
            self.file_name.append(gt_i['file_name'])
        self.fp = {}
        self.tp = {}
        self.sum_gt = {}
        for i in list(self.verb_name_dict.keys()):
            self.fp[i] = []
            self.tp[i] = []
            self.sum_gt[i] = 0
        self.num_class = len(list(self.verb_name_dict.keys()))

    def evalution(self, prediction_json):
        predict_annot = json.load(open(prediction_json, 'r'))
        for pred_i in predict_annot:
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            pred_bbox = pred_i['predictions']
            bbox_pairs = self.compute_iou_mat(gt_bbox, pred_bbox)
            pred_hoi = pred_i['hoi_prediction']
            gt_hoi = gt_i['hoi_annotation']
            self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs)
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        for i in list(self.verb_name_dict.keys()):

            sum_gt = self.sum_gt[i]
            if sum_gt == 0:
                continue
            tp = (self.tp[i]).copy()
            fp = (self.fp[i]).copy()
            res_num = len(tp)
            rec = np.zeros(res_num)
            prec = np.zeros(res_num)
            for v in range(res_num):
                if v > 0:
                    tp[v] = tp[v] + tp[v - 1]
                    fp[v] = fp[v] + fp[v - 1]
                rec[v] = tp[v] / sum_gt
                prec[v] = tp[v] / (tp[v] + fp[v])
            for v in range(res_num - 2, -1, -1):
                prec[v] = max(prec[v], prec[v + 1])
            for v in range(res_num):
                if v == 0:
                    ap[i] += rec[v] * prec[v]
                else:
                    ap[i] += (rec[v] - rec[v - 1]) * prec[v]
            max_recall[i] = np.max(rec)
            print('class {} --- ap: {}   max recall: {}'.format(
                i, ap[i], max_recall[i]))
        mAP = np.mean(ap[1:])
        m_rec = np.mean(max_recall[1:])
        print('--------------------')
        print('mAP: {}   max recall: {}'.format(mAP, m_rec))
        print('--------------------')
        return mAP

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        for gt_hoi_i in gt_hoi:
            if isinstance(gt_hoi_i['category_id'], str):
                gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n',''))
            if gt_hoi_i['category_id'] in list(self.verb_name_dict.keys()):
                self.sum_gt[gt_hoi_i['category_id']] += 1
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i['object_id'] in pos_pred_ids:
                    pred_dict_i = {'subject_id': match_pairs[pred_hoi_i['subject_id']], 'object_id': match_pairs[pred_hoi_i['object_id']], 'category_id': pred_hoi_i['category_id']}
                    if pred_dict_i in gt_hoi and vis_tag[gt_hoi.index(pred_dict_i)] == 0:
                        is_match = 1
                        vis_tag[gt_hoi.index(pred_dict_i)] = 1
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                if is_match == 1:
                    self.fp[pred_hoi_i['category_id']].append(0)
                    self.tp[pred_hoi_i['category_id']].append(1)

                else:
                    self.fp[pred_hoi_i['category_id']].append(1)
                    self.tp[pred_hoi_i['category_id']].append(0)

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        for i, bbox1 in enumerate(bbox_list1):
            max_iou = -0.1
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                if iou_i > max_iou:
                    iou_mat[i, j] = iou_i
                    max_iou = iou_i
        iou_mat[iou_mat>= 0.5] = 1
        iou_mat[iou_mat< 0.5] = 0
        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[0]):
                match_pairs_dict[pred_id] = match_pairs[1][i]
        return match_pairs_dict

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = rec1[2] * rec1[3]
            S_rec2 = rec2[2] * rec2[3]

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3] + rec1[1], rec2[3] + rec2[1])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2] + rec1[0], rec2[2] + rec1[0])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                return intersect / (sum_area - intersect)
        else:
            return 0


