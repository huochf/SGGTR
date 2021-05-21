import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def plot_prediction_results(cfg, images, targets, results, epoch_idx, iteration, ind_to_classes, ind_to_predicates, output_dir):
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    show_image = images.tensors[0].cpu().numpy()
    show_image = (show_image.transpose(1, 2, 0) * np.array(cfg.INPUT.PIXEL_STD) + np.array(cfg.INPUT.PIXEL_MEAN)) * 255 # here we support std are all zero by default
    show_image = show_image.astype(np.uint8)
    plt.imshow(show_image)

    colors = [(r, g, b) for r in [64., 128., 196.] for g in [64., 128., 196.] for b in [64., 128., 196.]]

    pred_boxes = results['graph_obj_box_lists'][0]
    keep = pred_boxes.get_field("keep").detach().cpu().numpy()
    boxes = pred_boxes.bbox.detach().cpu().numpy()
    labels = pred_boxes.get_field("pred_labels").detach().cpu().numpy()
    scores = pred_boxes.get_field("pred_scores").detach().cpu().numpy()
    rel_pair_idxs = pred_boxes.get_field("rel_pair_idxs").detach().cpu().numpy()
    # rel_pair_idxs = pred_boxes.get_field("vis_rel_pair_idxs").detach().cpu().numpy()
    # for i in range(boxes.shape[0]):
    #     box = boxes[i]
    #     label_name = ind_to_classes[labels[i]]
    #     color = np.array(colors[i % len(colors)]) / 255
    #     plt.gca().add_patch(
    #         plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
    #             fill=False, edgecolor=color, linewidth=1)
    #     )
    #     plt.text(box[0], box[1] - 2, label_name, fontsize=4, family="serif", 
    #         bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'),
    #         color='white')

    for i in range(rel_pair_idxs.shape[0]):
        sub_idx, obj_idx, rel_label = rel_pair_idxs[i]
        sub_box = boxes[sub_idx]
        obj_box = boxes[obj_idx]
        sub_label = labels[sub_idx]
        obj_label = labels[obj_idx]

        sub_name = ind_to_classes[sub_label]
        obj_name = ind_to_classes[obj_label]
        rel_name = ind_to_predicates[rel_label]
        
        sub_center = [(sub_box[2] + sub_box[0]) / 2, (sub_box[3] + sub_box[1]) / 2]
        obj_center = [(obj_box[2] + obj_box[0]) / 2, (obj_box[3] + obj_box[1]) / 2]
        rel_center = [(sub_center[0] + obj_center[0]) / 2, (sub_center[1] + obj_center[1]) / 2]

        color = np.array(colors[i % len(colors)]) / 255

        plt.gca().add_patch(
            plt.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1],
                fill=False, edgecolor=color, linewidth=1)
        )
        plt.gca().add_patch(
            plt.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1],
                fill=False, edgecolor=color, linewidth=1)
        )
        plt.gca().add_line(Line2D([sub_center[0], rel_center[0]], [sub_center[1], rel_center[1]], linewidth=1, color=color))
        plt.gca().add_line(Line2D([rel_center[0], obj_center[0]], [rel_center[1], obj_center[1]], linewidth=1, color=color))

        plt.plot(sub_center[0], sub_center[1], color=color, marker='.')
        plt.plot(obj_center[0], obj_center[1], color=color, marker='.')
        plt.plot(rel_center[0], rel_center[1], color=color, marker='.')
        plt.text(sub_center[0], sub_center[1] - 2, sub_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')
        plt.text(obj_center[0], obj_center[1] - 2, obj_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')
        plt.text(rel_center[0], rel_center[1] - 2, rel_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')


    plt.axis('off')
    ext = "png"
    if not os.path.exists(os.path.join(output_dir, "vis_images")):
        os.makedirs(os.path.join(output_dir, "vis_images"))
    output_path = os.path.join(output_dir, "vis_images", '%03d_%04d_pred.%s' % (epoch_idx, iteration, ext))
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.show()
    plt.close("all")

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    show_image = images.tensors[0].cpu().numpy()
    show_image = (show_image.transpose(1, 2, 0) * np.array(cfg.INPUT.PIXEL_STD) + np.array(cfg.INPUT.PIXEL_MEAN)) * 255 # here we support std are all zero by default
    show_image = show_image.astype(np.uint8)
    plt.imshow(show_image)
    target_boxlist = targets[0]

    colors = [(r, g, b) for r in [128., 196.] for g in [128., 196.] for b in [128., 196.]]
        
    boxes = target_boxlist.bbox.cpu().numpy()
    labels = target_boxlist.get_field("labels").cpu().numpy()
    rel_pair_idxs = target_boxlist.get_field("relation_tuple").cpu().numpy()
    # for i in range(boxes.shape[0]):
    #     box = boxes[i]
    #     label_name = ind_to_classes[labels[i]]
    #     color = np.array(colors[i % len(colors)]) / 255
    #     plt.gca().add_patch(
    #         plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
    #             fill=False, edgecolor=color, linewidth=1)
    #     )
    #     plt.text(box[0], box[1] - 2, label_name, fontsize=4, family="serif", 
    #         bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'),
    #         color='white')

    for i in range(rel_pair_idxs.shape[0]):
        sub_idx, obj_idx, rel_label = rel_pair_idxs[i]
        sub_box = boxes[sub_idx]
        obj_box = boxes[obj_idx]
        sub_label = labels[sub_idx]
        obj_label = labels[obj_idx]

        sub_name = ind_to_classes[sub_label]
        obj_name = ind_to_classes[obj_label]
        rel_name = ind_to_predicates[rel_label]

        sub_center = [(sub_box[2] + sub_box[0]) / 2, (sub_box[3] + sub_box[1]) / 2]
        obj_center = [(obj_box[2] + obj_box[0]) / 2, (obj_box[3] + obj_box[1]) / 2]
        rel_center = [(sub_center[0] + obj_center[0]) / 2, (sub_center[1] + obj_center[1]) / 2]

        color = np.array(colors[i % len(colors)]) / 255
        plt.gca().add_patch(
            plt.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1],
                fill=False, edgecolor=color, linewidth=1)
        )
        plt.gca().add_patch(
            plt.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1],
                fill=False, edgecolor=color, linewidth=1)
        )

        plt.gca().add_line(Line2D([sub_center[0], rel_center[0]], [sub_center[1], rel_center[1]], linewidth=1, color=color))
        plt.gca().add_line(Line2D([rel_center[0], obj_center[0]], [rel_center[1], obj_center[1]], linewidth=1, color=color))

        plt.plot(sub_center[0], sub_center[1], color=color, marker='.')
        plt.plot(obj_center[0], obj_center[1], color=color, marker='.')
        plt.plot(rel_center[0], rel_center[1], color=color, marker='.')
        plt.text(sub_center[0], sub_center[1] - 2, sub_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')
        plt.text(obj_center[0], obj_center[1] - 2, obj_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')
        plt.text(rel_center[0], rel_center[1] - 2, rel_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')


    plt.axis('off')
    ext = "png"
    if not os.path.exists(os.path.join(output_dir, "vis_images")):
        os.makedirs(os.path.join(output_dir, "vis_images"))
    output_path = os.path.join(output_dir, "vis_images", '%03d_%04d_gt.%s' % (epoch_idx, iteration, ext))
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.show()
    plt.close("all")



def _plot_prediction_results(cfg, images, targets, results, epoch_idx, iteration, ind_to_classes, ind_to_predicates, output_dir):

    show_image = images.tensors[0].cpu().numpy()
    show_image = (show_image.transpose(1, 2, 0) * np.array(cfg.INPUT.PIXEL_STD) + np.array(cfg.INPUT.PIXEL_MEAN)) * 255 # here we support std are all zero by default
    show_image = show_image.astype(np.uint8)

    colors = [(r, g, b) for r in [128., 255.] for g in [128., 255.] for b in [128., 255.]]

    fig = plt.figure(frameon=False, constrained_layout=False, dpi=300)

    if "box_lists" in results and cfg.LOSS.OD_ONLY:
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis('off')
        ax1.imshow(show_image)
        target = targets[0]
        plot_bbox(ax1, target, ind_to_classes, colors)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axis('off')
        ax2.imshow(show_image)
        result = results["box_lists"][0]
        plot_bbox(ax2, result, ind_to_classes, colors)

    elif "box_lists" in results and "graph_obj_box_lists" in results and not cfg.LOSS.SGG_ONLY:
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis('off')
        ax1.imshow(show_image)
        target = targets[0]
        plot_bbox(ax1, target, ind_to_classes, colors)
        plot_relation(ax1, target, ind_to_classes, ind_to_predicates, colors)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis('off')
        ax2.imshow(show_image)
        result = results["box_lists"][0]
        plot_bbox(ax2, result, ind_to_classes, colors)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.axis("off")
        ax3.imshow(show_image)
        result = results["graph_obj_box_lists"][0]
        plot_bbox(ax3, result, ind_to_classes, colors)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        ax4.imshow(show_image)
        # plot_bbox(ax4, result, ind_to_classes, colors)
        plot_relation(ax4, result, ind_to_classes, ind_to_predicates, colors)

    elif "rel_box_lists" in results or "graph_obj_box_lists" in results:
        ax3 = fig.add_subplot(1, 3, 1)
        ax3.axis("off")
        ax3.imshow(show_image)
        target = targets[0]
        plot_bbox(ax3, target, ind_to_classes, colors)
        plot_relation(ax3, target, ind_to_classes, ind_to_predicates, colors)

        ax1 = fig.add_subplot(1, 3, 2)
        ax1.axis("off")
        ax1.imshow(show_image)
        result = results["graph_obj_box_lists"][0] if "graph_obj_box_lists" in results else results["rel_box_lists"][0]
        plot_bbox(ax1, result, ind_to_classes, colors)

        ax4 = fig.add_subplot(1, 3, 3)
        ax4.axis("off")
        ax4.imshow(show_image)
        # plot_bbox(ax4, result, ind_to_classes, colors)
        plot_relation(ax4, result, ind_to_classes, ind_to_predicates, colors)

    ext = "png"
    if not os.path.exists(os.path.join(output_dir, "vis_images")):
        os.makedirs(os.path.join(output_dir, "vis_images"))
    output_path = os.path.join(output_dir, "vis_images", str(epoch_idx) + '_' + str(iteration) + '.' + ext)
    fig.savefig(output_path, dpi=300)
    plt.close("all")


def plot_bbox(ax, box_list, ind_to_classes, colors):
    areas = box_list.area()
    sorted_inds = np.argsort(-areas.detach().cpu().numpy())
    boxes = box_list.bbox.detach().cpu().numpy()
    labels = box_list.get_field("labels").detach().cpu().numpy() if box_list.has_field("labels") else box_list.get_field("pred_labels").detach().cpu().numpy()
    scores = box_list.get_field("pred_scores").detach().cpu().numpy() if box_list.has_field("pred_scores") else None

    if box_list.has_field("keep"):
        keep = box_list.get_field("keep").detach().cpu().numpy()
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        sorted_inds = np.argsort(-areas.detach().cpu().numpy()[keep])

    for i in sorted_inds[:100]:
        box = boxes[i]
        label_name = ind_to_classes[labels[i]]
        if scores is not None:
            label_name = label_name + '[{:.2f}]'.format(scores[i])
        color = np.array(colors[i % len(colors)]) / 255

        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                fill=False, edgecolor=color, linewidth=1)
        )
        ax.text(box[0], box[1] - 2, label_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'),
            color='white')


def plot_relation(ax, box_list, ind_to_classes, ind_to_predicates, colors):
    boxes = box_list.bbox.detach().cpu().numpy()
    labels = box_list.get_field("labels").detach().cpu().numpy() if box_list.has_field("labels") else box_list.get_field("pred_labels").detach().cpu().numpy()
    rel_pair_idxs = box_list.get_field("rel_pair_idxs").detach().cpu().numpy() if box_list.has_field("rel_pair_idxs") else box_list.get_field("relation_tuple").detach().cpu().numpy()
    pred_rel_scores = box_list.get_field("pred_rel_scores").detach().cpu().numpy() if box_list.has_field("pred_rel_scores") else None

    if box_list.has_field("vis_rel_pair_idxs"):
        rel_pair_idxs = box_list.get_field("vis_rel_pair_idxs")
        pred_rel_scores = box_list.get_field("vis_pred_rel_scores")

    for i in range(min(rel_pair_idxs.shape[0], 100)):
        sub_idx, obj_idx, rel_label = rel_pair_idxs[i]
        sub_box = boxes[sub_idx]
        obj_box = boxes[obj_idx]
        sub_label = labels[sub_idx]
        obj_label = labels[obj_idx]

        sub_name = ind_to_classes[sub_label]
        obj_name = ind_to_classes[obj_label]
        rel_name = ind_to_predicates[rel_label]

        sub_center = [(sub_box[2] + sub_box[0]) / 2, (sub_box[3] + sub_box[1]) / 2]
        obj_center = [(obj_box[2] + obj_box[0]) / 2, (obj_box[3] + obj_box[1]) / 2]
        rel_center = [(sub_center[0] + obj_center[0]) / 2, (sub_center[1] + obj_center[1]) / 2]

        color = np.array(colors[i % len(colors)]) / 255

        ax.add_patch(
            plt.Rectangle((sub_box[0], sub_box[1]), sub_box[2] - sub_box[0], sub_box[3] - sub_box[1],
                fill=False, edgecolor=color, linewidth=1)
        )
        ax.add_patch(
            plt.Rectangle((obj_box[0], obj_box[1]), obj_box[2] - obj_box[0], obj_box[3] - obj_box[1],
                fill=False, edgecolor=color, linewidth=1)
        )

        ax.add_line(Line2D([sub_center[0], rel_center[0]], [sub_center[1], rel_center[1]], linewidth=1, color=color))
        ax.add_line(Line2D([rel_center[0], obj_center[0]], [rel_center[1], obj_center[1]], linewidth=1, color=color))
        
        ax.plot(sub_center[0], sub_center[1], color=color, marker='.')
        ax.plot(obj_center[0], obj_center[1], color=color, marker='.')
        ax.plot(rel_center[0], rel_center[1], color=color, marker='.')
        ax.text(sub_center[0], sub_center[1] - 2, sub_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')
        ax.text(obj_center[0], obj_center[1] - 2, obj_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')
        ax.text(rel_center[0], rel_center[1] - 2, rel_name, fontsize=4, family="serif", 
            bbox=dict(facecolor=color, alpha=1.0, pad=0, edgecolor='none'), color='white')
