#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   v1_table_house_sale_confirm_order.py
@Time    :   2020/08/31 17:33:21
@Author  :   lzneu
@Version :   1.0
@Contact :   lizhuang009@ke.com
@License :   (C)Copyright 2020-2021,KeOCR
@Desc    :   新房成销单：表格、字段 增强后处理代码
'''

import os
import json
import pickle
import base64
from math import sqrt
from flask import g
from app.libs import utils
from flask import current_app
from app.processes.base import BaseProcess
import re
from app.models.detect.lcnn.tools import line2frame
# from app.models.detect.lcnn.tools.zx_tools import OneTable
from shapely.geometry import Polygon
import numpy as np
import copy


class PostProcessingRows(object):
    """
    按行返回后处理代码
    """

    def __init__(self, pkl_data):
        self.__data = dict(
            sorted(pkl_data.items(), key=lambda x: x[1][0][1]))  # 按照y1进行升序排列

    def get_rows(self):

        boxes = sorted(self.__data.values(), key=lambda x: x[0][1])  # 按y排序
        num_box = len(boxes)

        final_result = []
        row = []
        flag = True
        i = 1
        while i < num_box:
            j = i - 1
            box_1 = boxes[j]
            if flag:
                row.append(box_1)
            box_1_width = box_1[0][7] - box_1[0][1] - 2  # 增加2个像素的padding
            box_2 = boxes[i]
            gap = box_2[0][1] - box_1[0][1]
            gap_x = min(abs(box_2[0][2]-box_1[0][0]), abs(box_2[0][0]-box_1[0][2]))

            if gap < box_1_width and gap_x < 128:  # 四个字符？
                row.append(box_2)
                flag = False
                i += 1
            else:
                final_result.append(row)
                row = []
                flag = True
                i += 1
                if i >= num_box:
                    row = [box_2]
        # 处理最后一行
        if row != []:
            final_result.append(row)
        # 排序行中的文本
        rows = []
        for box_list in final_result:
            box_list = sorted(box_list, key=lambda x: x[0][0])
            word_list = [box[1] for box in box_list]
            prop_row = "".join(word_list)
            # 甲乙方合在一起的情况特殊处理
            if len(word_list) == 4 and '甲方' in prop_row and '乙方' in prop_row:
                box_list = sorted(box_list, key=lambda x: x[0][1])
                jyf_value_list = [box[1] for box in box_list if "甲方" not in box[1] and "乙方" not in box[1]]
                if len(jyf_value_list) >= 2:
                    rows.append('甲方'+jyf_value_list[0])
                    rows.append('乙方'+jyf_value_list[1])
                    continue
            row_content = ''
            for word in word_list:
                flg = False
                for key_v in ["明细表", "确认单", "确认书", "对账单", "结算单", "确认表", "结算佣金", "成销表", "结算单", "对帐单", "进度表", "明细"]:
                    if key_v in word:
                        rows.append(word)
                        flg = True
                        break
                if not flg:
                    row_content += word
            rows.append(row_content)
        return rows


class Process(BaseProcess):
        
    
    def process(self):
        """
        新房成销单后处理代码
        :param ret: dict类型
            ret['oneTable_list_all']:  oneTable对象的list，用来存储表格结构
            ret['recong_data']: 文字OCR结果
        :return res: dict类型，包含4个字段
        """
        self.result = {
        "XIANG_MU_MING_CHENG": "",      # str
        "JIA_FANG": "",                 # str
        "YI_FANG": "",                  # str
        "OUT_WORDS": [],
        "TABLE_INFO": "",
        "COORDINATES": {
            "XIANG_MU_MING_CHENG": [],
            "JIA_FANG": [],
            "YI_FANG": []
        },
        "RECOGNISE_AREA_COORDINATES" : []
        }
        # self.drop_words = ['确认', '盖章', '签字']
        # 获取数据
        #  self.params 存储了所有识别的数据
        oneTable_list = self.params['one_tables']
        rows = PostProcessingRows(self.boxes).get_rows()
        # 这里对甲乙方做个处理
        rows = self.deal_jiayifang(rows, self.boxes)

        # 1、从所有文字中提取四个字段
        # 表格外部的所有文本 返回一个list ，从上到下进行排序
        table_loc = [[], [], [], []]
        self.result['OUT_WORDS'] = []
        self.drop_words = ['盖章', '名称', '签字', '确认', '合同']
        for index, content in enumerate(rows):
            """结构化提取"""
            # 当前识别模型的强揪
            content = content.replace("朝细表", "明细表")
            content = content.replace("里方", "甲方")

            # 项目名称:不论在表格内还是表格外，统一返回标题，一般包含如下字段：成功销售确认单、成销确认书、成交确认书、结算单、对账单
            if len(self.result['XIANG_MU_MING_CHENG']) == 0:
                for key_v in ["明细表", "确认单", "确认书", "对账单", "结算单", "确认表", "结算佣金", "成销表", "结算单", "对帐单", "进度表", "明细"]:
                    if key_v in content:
                        if len(content.split(key_v)[0]) > 0:
                            content = content.split(key_v)[0] + key_v
                        self.result['XIANG_MU_MING_CHENG'] = content 
                        break
                                         
            # 甲方:同一表述：合同主体：甲方、甲方名称、合作方公司、合作方公司名称
            if len(self.result['JIA_FANG']) == 0 and "甲方" in content:
                    jf_content = self._patstr(content.split("甲方")[-1], r"、()（）a-zA-Z0-9\u4e00-\u9fa5")
                    for w in self.drop_words:
                        jf_content = jf_content.replace(w, '')
                    self.result['JIA_FANG'] = jf_content
            # 乙方:同一表述：合同主体：乙方：、乙方名称
            if len(self.result['YI_FANG']) == 0 and "乙方" in content:
                yf_content = self._patstr(content.split("乙方")[-1], r"、()（）a-zA-Z0-9\u4e00-\u9fa5")
                for w in self.drop_words:
                    yf_content = yf_content.replace(w, '')
                # 只保留汉字、数字、字母
                self.result['YI_FANG'] = yf_content

        # 2、将oneTable变成excel oneTable是decode之后的对象
        # 存储目录
        ocr_img_dir = g.ocr_img_dir
        for index, oneTable in enumerate(oneTable_list):
            table_cell_list = oneTable.table_cell_list
            img_name = 'table-' +str(index)+'.xlsx'
            excel_path = os.path.join(ocr_img_dir, img_name)
            line2frame.MergeExcelWriteData(table_cell_list, excel_path).merge_excel()
            with open(excel_path, 'rb') as fh:
                excel_data = fh.read()
                excel_data = base64.b64encode(excel_data).decode('utf-8')
                self.result["TABLE_INFO"] = excel_data
                
            # 先计算行数、 列数
            row_num = 0
            col_num = 0
            y1 = 0
            y2 = 0
            for table_cell in table_cell_list:
                row_num = max(row_num, table_cell.end_row)
                col_num = max(col_num, table_cell.end_col)
            for table_cell in table_cell_list:
                if table_cell.st_col == 0:
                    y1 = max(y1, table_cell.loc[7])
                if table_cell.end_col == col_num:
                    y2 = max(y2, table_cell.loc[5])

            # 求表格边界
            ratio = max(oneTable.width, oneTable.height) / 512.0
            # 加上params['target_boxes']的左上角顶点坐标 放大到原始图片
            target_box = self.params['target_boxes'][index]
            x, y = target_box[0], target_box[1]
            table_loc =  [0, 0, oneTable.width+x, 0, oneTable.width+x, y2*ratio+y, 0+x, y1*ratio+y]
            # 画目标检测的图 如果不需要可以覆盖
            # rotate_name = current_app.config['ROTATE_PIC']
            # utils.demo_draw(rotate_name, [table_loc])
            # 开始求外面的文本
            out_boxes = {}
            for index, v in self.boxes.items():
                loc, rec = v[0], v[1]
                if not self.is_inter(table_loc, loc):
                    out_boxes[index] = v

            # 合并
            out_rows = PostProcessingRows(out_boxes).get_rows()
            self.result['OUT_WORDS'] = out_rows
            break  # 因为新房成销单全部为单表格            
    
        return self.result

    def is_inter(self, a, b):
        """
        两个多边形是否相交
        :param
        :return
        """
        a = np.array(a).reshape(-1, 2)
        b = np.array(b).reshape(-1, 2) 
        poly1 = Polygon(a).convex_hull 
        poly2 = Polygon(b).convex_hull 
        inter_area = poly1.intersection(poly2).area 
        if inter_area > 0.3:
            return True
        else:
            return False
    def deal_jiayifang(self, rows, boxes):
        jf_flg = False
        yf_flg = False
        # 只有当二者第一次出现，且，以关键字结尾才开始处理
        for i, row in enumerate(rows):
            if not yf_flg and '乙方' in row:
                yf_flg = True
                if row.endswith('乙方') or row.endswith('乙方:') or row.endswith('乙方：'):
                    # 这里说明乙方没有匹配到，使用boxes进行匹配
                    yf_name = ""
                    for j, box in boxes.items():
                        loc, content, _ = box
                        if row.endswith(content):
                            # 说明这个文本检测框位置
                            y_min = (loc[3] + loc[5]) / 2
                            x_min = loc[2]
                            # 在boxes中找到loc[1]-loc[7]的x最小的box
                            boxes_bak = copy.deepcopy(boxes).values()
                            chose_boxes = sorted(boxes_bak, key= lambda x: (x[0][0]+x[0][6]))
                            for chose_box in chose_boxes:
                                if chose_box[0][1] <= y_min and chose_box[0][7] >= y_min and content != chose_box[1] and chose_box[0][0] > x_min:
                                    yf_name = chose_box[1]
                                    if '甲方' in yf_name:
                                        yf_name = ''
                                    break
                            break
                    rows[i] = row + yf_name
            if not jf_flg and '甲方' in row:
                jf_flg = True
                if row.endswith('甲方') or row.endswith('甲方:') or row.endswith('甲方：'):
                    # 这里说明甲方没有匹配到，使用boxes进行匹配
                    jf_name = ""
                    for j, box in boxes.items():
                        loc, content, _ = box
                        if row.endswith(content):
                            # 说明这个文本检测框位置
                            y_min = (loc[3] + loc[5]) / 2
                            x_min = loc[2]
                            # 在boxes中找到loc[1]-loc[7]的x最小的box
                            boxes_bak = copy.deepcopy(boxes).values()
                            chose_boxes = sorted(boxes_bak, key= lambda x: (x[0][0]+x[0][6]))
                            for chose_box in chose_boxes:
                                if chose_box[0][1] <= y_min and chose_box[0][7] >= y_min and content != chose_box[1] and chose_box[0][0] > x_min:
                                    jf_name = chose_box[1]
                                    if '乙方' in jf_name:
                                        jf_name = ''
                                    break
                            break
                    rows[i] = row + jf_name
        return rows
        

if __name__ == "__main__":
    pass
