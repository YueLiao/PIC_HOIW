import json
import os, glob
import cv2
import numpy as np
from graphviz import Digraph
class visualize():
    def __init__(self,image_dir, color_map = None,line_color_map = None, is_put_text = False, pad_size = (0, 0)):
        self.is_put_text = is_put_text
        self.image_dir = image_dir
        self.pad_size = pad_size
        self.text_color = (0, 0, 255)
        if color_map == None:
            self.bbox_color_map = np.load('colormap.npy')
        else:
            self.bbox_color_map = color_map
        if line_color_map == None:
            self.line_color_map = np.load('line_color_map.npy')
        else:
            self.line_color_map = line_color_map
        self.annotations = None
        self.mode = None
        self.file_name = []
        self.sub_color = (0, 0, 255)
        self.obj_color = (0, 255, 0)
    def vis_bbox(self, image, bbox_list):
        pad_img = np.pad(image, (self.pad_size, self.pad_size, (0, 0)), 'constant', constant_values=128)
        for bbox in bbox_list:
            bbox_cor = bbox['bbox'].copy()
            if self.bbox_mode == 'xywh':
                bbox_cor[2] = bbox_cor[2] + bbox_cor[0]
                bbox_cor[3] = bbox_cor[3] + bbox_cor[1]
            if 'category_id' in bbox.keys() and  self.mode != 'part_bbox':
                bbox_cate = int(bbox['category_id'])
                this_color_map = (np.asscalar(self.bbox_color_map[bbox_cate][0]), np.asscalar(self.bbox_color_map[bbox_cate][1]),
                              np.asscalar(self.bbox_color_map[bbox_cate][2]))
            else:
                this_color_map = self.bbox_color_map[0]
            cv2.rectangle(pad_img, (bbox_cor[0] + self.pad_size[0], bbox_cor[1] + self.pad_size[1]), (bbox_cor[2] + self.pad_size[0], bbox_cor[3] + self.pad_size[1]),
                          this_color_map, 2)

            if self.is_put_text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(pad_img, 'bbox_id:' + str(bbox['id']), (bbox_cor[0] + self.pad_size[0] - 5, bbox_cor[1] + self.pad_size[1] - 5), font, 1, self.text_color, 2)
        return pad_img

    def vis_hoi(self, image, bbox_list, hoi_list):
        vis_bbox_img = self.vis_bbox(image, bbox_list)
        dot = Digraph(format='png')
        for id, bbox in enumerate(bbox_list):
            if bbox['category_id'] == '1':
                node_color = 'red'
            else:
                node_color = 'green'
            dot.node(str(id), self.obj_name_dict[int(bbox['category_id'])] + ' ' + str(id), color = node_color)
        for hoi in hoi_list:
            dot.edge(str(hoi[0]), str(hoi[1]), self.verb_name_dict[hoi[2]])
            if hoi[0] >= len(bbox_list) or hoi[1] >= len(bbox_list):
                continue
            sub_bbox = bbox_list[hoi[0]]['bbox']
            obj_bbox = bbox_list[hoi[1]]['bbox']
            hoi_cate = hoi[2]
            this_line_color_map = (
            np.asscalar(self.line_color_map[hoi_cate][0]), np.asscalar(self.line_color_map[hoi_cate][1]),
            np.asscalar(self.line_color_map[hoi_cate][2]))

            if self.bbox_mode == 'xywh':
                sub_loc = (int(sub_bbox[0] + sub_bbox[2] / 2 + self.pad_size[0]),
                                        int(sub_bbox[1] + sub_bbox[3] / 2 + self.pad_size[1]))
                obj_loc = (int((obj_bbox[0] + obj_bbox[2] / 2) + self.pad_size[0]),
                          int((obj_bbox[1] + obj_bbox[3] / 2) + self.pad_size[1]))
                cv2.line(vis_bbox_img, sub_loc, obj_loc, this_line_color_map,3)
                cv2.circle(vis_bbox_img, sub_loc, 3, self.sub_color, -1)
                cv2.circle(vis_bbox_img, obj_loc, 3, self.obj_color, -1)

            elif self.bbox_mode == 'xyxy':
                sub_loc = (int((sub_bbox[0] + sub_bbox[2]) / 2 + self.pad_size[0]),
                                        int((sub_bbox[1] + sub_bbox[3]) / 2 + self.pad_size[1]))
                obj_loc = (int((obj_bbox[0] + obj_bbox[2]) / 2 + self.pad_size[0]),
                          int((obj_bbox[1] + obj_bbox[3]) / 2 + self.pad_size[1]))
                cv2.line(vis_bbox_img, sub_loc, obj_loc, this_line_color_map, 3)
                cv2.circle(vis_bbox_img, sub_loc, 3, self.sub_color, -1)
                cv2.circle(vis_bbox_img, obj_loc, 3, self.obj_color, -1)
        dot.render('temp')
        vis_hoi_img = cv2.imread('temp.png')
        dst = (int(vis_bbox_img.shape[0] / float(vis_hoi_img.shape[0]) * vis_hoi_img.shape[1]), vis_bbox_img.shape[0])
        vis_out = np.concatenate((vis_bbox_img, cv2.resize(vis_hoi_img, dst)), 1)
        return vis_out

    def load_annot(self, annot_file, mode = 'iccv_hoiw'):
        self.mode = mode
        if self.mode == 'iccv_hoiw':
            self.bbox_mode = 'xyxy'
            self.verb_name_dict = {1: 'smoke', 2: 'call', 3: 'play(cellphone)', 4: 'eat', 5: 'drink',
                                   6: 'ride', 7: 'hold', 8: 'kick', 9: 'read', 10: 'play (computer)'}
            self.obj_name_dict = {1: 'person', 2: 'telephone', 3: 'cigarette', 4: 'drink', 5: 'food',
                                  6: 'cycle', 7: 'motorbike', 8: 'horse', 9: 'ball', 10: 'document', 11: 'computer'}
            format_annot = []
            annotation = json.load(open(annot_file, 'r'))
            for h_annot in annotation:
                if h_annot.__contains__('hoi_annotation'):
                    hoi_annt = [[h['subject_id'], h['object_id'], h['category_id']] for
                                h in h_annot['hoi_annotation']]
                    new_annot = {'bbox_list':h_annot['annotations'], 'hoi': hoi_annt}
                    h_annot.pop('hoi_annotation')
                    h_annot['annotations'] = new_annot
                self.file_name.append(h_annot['file_name'])
                format_annot.append(h_annot)
            self.annotations = format_annot
        else:
            raise NotImplementedError

    def vis_one_image(self, filename):
        image = cv2.imread(os.path.join(self.image_dir, filename))
        annotation = self.annotations[self.file_name.index(filename)]
        print(annotation)
        if annotation['annotations'].__contains__('hoi'):
            bbox_list = annotation['annotations']['bbox_list']
            hoi = annotation['annotations']['hoi']
            vis_img = self.vis_hoi(image, bbox_list, hoi)
        else:
            bbox_list = annotation
            vis_img = self.vis_bbox(image, bbox_list)
        cv2.imshow('vis img', vis_img)
        cv2.waitKey(0)

    def vis_all_images(self, outdir):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for annot in self.annotations:

            image = cv2.imread(os.path.join(self.image_dir,annot['file_name']))
            if annot['annotations'].__contains__('hoi'):
                bbox_list = annot['annotations']['bbox_list']
                if len(bbox_list) < 2:
                    print(annot['file_name'])
                    continue
                hoi = annot['annotations']['hoi']
                vis_img = self.vis_hoi(image, bbox_list, hoi)
            else:
                bbox_list = annot['annotations']
                vis_img = self.vis_bbox(image, bbox_list)
            cv2.imwrite(os.path.join(outdir, annot['file_name']), vis_img)
