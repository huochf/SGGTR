# Modified from Scene-Graph-Benchmark (https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
# ----------------------------------------------------------------------------------------------------
import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .sgg_eval import (
    SGRecall, 
    SGNoGraphConstraintRecall, 
    SGZeroShotRecall, 
    SGNGZeroShotRecall, 
    SGPairAccuracy, 
    SGMeanRecall, 
    SGNGMeanRecall, 
    SGAccumulateRecall
)

def do_vrd_evaluation(cfg, epoch, dataset, predictions, output_folder, logger, iou_types):

    num_rel_category = cfg.MODEL.DETECTION_HEADER.NUM_REL_CLASS + 1
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD

    for field, prediction_per_field in predictions.items():
        groundtruths = []
        for image_id, prediction in enumerate(prediction_per_field):
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            # recover original size which is before transform
            prediction_per_field[image_id] = prediction.resize((image_width, image_height))

            gt = dataset.get_groundtruth(image_id, evaluation=True)
            groundtruths.append(gt)
        predictions[field] = prediction_per_field

    save_output(output_folder, groundtruths, predictions, dataset)

    result_str = "\n" + '=' * 100 + '\n'
    if 'bbox' in iou_types:
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field("labels").tolist() # integer
            boxes = gt.bbox.tolist() # xyxy
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1], # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                    })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(groundtruths))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name}
                for i, name in enumerate(dataset.ind_to_classes) if name != "__background__"
                ],
            "annotations": anns,
        }
        fauxcoco.createIndex()

        if "box_lists" in predictions:
            pred_box_lists = predictions["box_lists"]

            # format predictions to coco-like
            cocolike_predictions = []
            for image_id, prediction in enumerate(pred_box_lists):
                if len(prediction) < 1:
                    continue
                box = prediction.convert("xywh").bbox.detach().cpu().numpy() # xywh
                score = prediction.get_field("pred_scores").detach().cpu().numpy() # (#objs, )
                label = prediction.get_field("pred_labels").detach().cpu().numpy() # (#objs, )
                image_id = np.asarray([image_id] * len(box))
                cocolike_predictions.append(
                    np.column_stack((image_id, box, score, label))
                )

            if cocolike_predictions != []:
                cocolike_predictions = np.concatenate(cocolike_predictions, 0)

            if len(cocolike_predictions) < 1:
                res = COCO()
            else:
                # evaluate via coco API
                res = fauxcoco.loadRes(cocolike_predictions)
            logger.info("Begin evaluate object detection performance...")
            coco_eval = COCOeval(fauxcoco, res, 'bbox')
            coco_eval.params.imgIds = list(range(len(groundtruths)))
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAp = coco_eval.stats[1]

            result_str += "Object Detection evaluation mAp = %.4f\n" % mAp 
            result_str += '=' * 100 + '\n'

        if "graph_obj_box_lists" in predictions:
            pred_box_lists = predictions["graph_obj_box_lists"]

            # format predictions to coco-like
            cocolike_predictions = []
            for image_id, prediction in enumerate(pred_box_lists):
                if len(prediction) < 1:
                    continue
                box = prediction.convert("xywh").bbox.detach().cpu().numpy() # xywh
                score = prediction.get_field("pred_scores").detach().cpu().numpy() # (#objs, )
                label = prediction.get_field("pred_labels").detach().cpu().numpy() # (#objs, )
                image_id = np.asarray([image_id] * len(box))
                cocolike_predictions.append(
                    np.column_stack((image_id, box, score, label))
                )

            if cocolike_predictions != []:
                cocolike_predictions = np.concatenate(cocolike_predictions, 0)

            if len(cocolike_predictions) < 1:
                res = COCO()
            else:
                # evaluate via coco API
                res = fauxcoco.loadRes(cocolike_predictions)
            logger.info("Begin evaluate object detection performance...")
            coco_eval = COCOeval(fauxcoco, res, 'bbox')
            coco_eval.params.imgIds = list(range(len(groundtruths)))
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAp = coco_eval.stats[1]

            result_str += "Scene Graph Detection evaluation mAp = %.4f\n" % mAp 
            result_str += '=' * 100 + '\n'

    mode = "sgdet"
    if "relations" in iou_types:
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # used for no graph constraint mean Recall@K
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        # prepare all inputs
        global_container = {}
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = False
        global_container['num_attributes'] = -1
        
        if "rel_box_lists" in predictions:
            predictions = predictions["rel_box_lists"]
            # groundtruths = groundtruths["rel_box_lists"]
        else:
            assert "graph_obj_box_lists" in predictions
            predictions = predictions["graph_obj_box_lists"]
            # groundtruths = groundtruths["graph_obj_box_lists"]

        for groundtruth, prediction in zip(groundtruths, predictions):
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)
        
        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)
        eval_ng_mean_recall.calculate_mean_recall(mode)
        
        # print result
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_nog_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        result_str += eval_ng_mean_recall.generate_print_string(mode)
        
        result_str += '=' * 100 + '\n'

    logger.info(result_str)

    if "relations" in iou_types:
        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, str(epoch) + '_result_dict.pytorch'))
        return float(np.mean(result_dict[mode + '_recall'][100]))
    elif "bbox" in iou_types:
        return float(mAp)
    else:
        return -1


def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths': groundtruths, 'predictions': predictions}, os.path.join(output_folder, str(epoch) + "_eval_results.pytorch"))

        visual_info = {}
        for field, in groundtruth.keys():
            groundtruth_per_field = groundtruths[field]
            prediction_per_field = predictions[field]
            visual_info_per_field = []
            for image_id, (groundtruth, prediction) in enumerate(zip(groundtruth_per_field, prediction_per_field)):
                img_file = os.path.abspath(dataset.filenames[image_id])
                groundtruth = [
                    [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                    for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field("labels").tolist())
                ]
                prediction = [
                    [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                    for b, l in zip(prediction.bbox.tolist(), prediction.get_field("pred_labels").tolist())
                ]
                visual_info_per_field.append({
                    'img_file': img_file,
                    'groundtruth': groundtruth,
                    'prediction': prediction
                    })
            visual_info[field] = visual_info_per_field
        with open(os.path.join(output_folder, str(epoch) + "_visual_info.json"), "w") as f:
            json.dump(visual_info, f)


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    #unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()                   # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()           # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()[:, :2] # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field('rel_cls_scores').detach().cpu().numpy()          # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()                  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()     # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()              # (#pred_objs, )


    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # No Graph Constraint Mean Recall
    evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)

    return 
