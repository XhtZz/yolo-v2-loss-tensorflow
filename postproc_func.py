import numpy as np

def bbox_ious_numpy(boxes1, boxes2):
    b1_len = boxes1.shape[0]
    b2_len = boxes2.shape[0]

    boxes1_0 = np.reshape(boxes1[:, 0],(b1_len,1))
    boxes1_1 = np.reshape(boxes1[:, 1],(b1_len,1))
    boxes1_2 = np.reshape(boxes1[:, 2],(b1_len,1))
    boxes1_3 = np.reshape(boxes1[:, 3],(b1_len,1))

    boxes2_0 = np.reshape(boxes2[:, 0],(1,b2_len))
    boxes2_1 = np.reshape(boxes2[:, 1],(1,b2_len))
    boxes2_2 = np.reshape(boxes2[:, 2],(1,b2_len))
    boxes2_3 = np.reshape(boxes2[:, 3],(1,b2_len))

    dx_left = np.maximum(boxes1_0 - boxes1_2 / 2, boxes2_0 - boxes2_2 / 2)
    dx_right = np.minimum(boxes1_0 + boxes1_2 / 2, boxes2_0 + boxes2_2 / 2)

    dx_overlap = dx_right - dx_left
    dx_overlap = np.maximum(dx_overlap, np.zeros_like(dx_overlap))

    dy_left = np.maximum(boxes1_1 - boxes1_3 / 2, boxes2_1 - boxes2_3 / 2)
    dy_right = np.minimum(boxes1_1 + boxes1_3 / 2, boxes2_1 + boxes2_3 / 2)

    dy_overlap = dy_right - dy_left
    dy_overlap = np.maximum(dy_overlap, np.zeros_like(dy_overlap))

    intersections = dx_overlap * dy_overlap

    unions = boxes1_2 * boxes1_3 + boxes2_2 * boxes2_3 - intersections

    return intersections / np.maximum(unions, 1e-10)



def build_groundTrue_targets_masks(ground_truth, batch_size, anchors_num, nH, nW, thresh, reduction):
    anchors = np.array([(16, 16), (48, 48), (80, 80),(112,112),(160,160)])/reduction
    object_detections   = np.zeros((batch_size, anchors_num, nH*nW),dtype=float)
    object_no_detections= np.ones((batch_size, anchors_num, nH*nW),dtype=float)
    coord_mask          = np.zeros((batch_size, anchors_num, 1, nH*nW),dtype=float)
    gt_coord            = np.zeros((batch_size, anchors_num, 4, nH*nW),dtype=float)
    gt_conf             = np.zeros((batch_size, anchors_num, nH*nW),dtype=float)

    for b in range(batch_size):
        ground_truth_tensor = np.reshape(ground_truth[b],(-1,5))
        if np.sum(ground_truth_tensor) == 0:   # No gt for this image
            continue

        #do scaling, base on output feature map 
        imagesize_w = 304.0
        imagesize_h = 224.0
        xywh_scale  = np.array([1,imagesize_w,imagesize_h,imagesize_w,imagesize_h])
        gt          = ground_truth_tensor[:,1:5] * xywh_scale[1:5] / reduction

        # Find best anchor for each gt
        groundtruth_num = ground_truth_tensor.shape[0]
        gt_00wh         = np.zeros((groundtruth_num,4))
        gt_00wh[:,2:4]  = gt[:,2:4]

        anchors_num     = anchors.shape[0]
        anchors_00wh    = np.zeros((anchors_num,4))
        anchors_00wh[:,2:4] = anchors

        iou_gt_anchors     = bbox_ious_numpy(gt_00wh, anchors_00wh)
        bestanchors_index  = np.argmax(iou_gt_anchors,axis=1)

        # Set masks and target values for each gt
        for i in range(groundtruth_num):
            anno = ground_truth_tensor[i] * xywh_scale
            gi   = np.clip(int(gt[i, 0]), 0, nW-1)
            gj   = np.clip(int(gt[i, 1]), 0, nH-1)
            best_n = bestanchors_index[i]

            if anno[3] < 10 or anno[4] < 10:
                object_detections[b][best_n][gj * nW + gi] = 0
                object_no_detections[b][best_n][gj * nW + gi] = 0
            else:
                coord_mask[b][best_n][0][gj * nW + gi] = 2 - anno[3] * anno[4] / (
                            nW * nH * reduction * reduction)
                       
                object_detections[b][best_n][gj*nW+gi] = 1
                object_no_detections[b][best_n][gj*nW+gi] = 0

                gt_coord[b][best_n][0][gj*nW+gi] = gt[i, 0] - gi
                gt_coord[b][best_n][1][gj*nW+gi] = gt[i, 1] - gj
                gt_coord[b][best_n][2][gj*nW+gi] = np.log(gt[i, 2]/anchors[best_n, 0])
                gt_coord[b][best_n][3][gj*nW+gi] = np.log(gt[i, 3]/anchors[best_n, 1])

                gt_conf[b][best_n][gj*nW+gi] = 1

    return coord_mask, object_detections, object_no_detections, gt_coord, gt_conf