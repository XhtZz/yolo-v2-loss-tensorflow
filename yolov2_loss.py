import tensorflow as tf

def bbox_ious_tf(boxes1, boxes2):
    b1_len = boxes1.shape[0]
    b2_len = boxes2.shape[0]

    boxes1_0 = tf.reshape(boxes1[:, 0],[b1_len,1])
    boxes1_1 = tf.reshape(boxes1[:, 1],[b1_len,1])
    boxes1_2 = tf.reshape(boxes1[:, 2],[b1_len,1])
    boxes1_3 = tf.reshape(boxes1[:, 3],[b1_len,1])

    boxes2_0 = tf.reshape(boxes2[:, 0],[1,b2_len])
    boxes2_1 = tf.reshape(boxes2[:, 1],[1,b2_len])
    boxes2_2 = tf.reshape(boxes2[:, 2],[1,b2_len])
    boxes2_3 = tf.reshape(boxes2[:, 3],[1,b2_len])

    dx_left = tf.maximum(boxes1_0 - boxes1_2 / 2, boxes2_0 - boxes2_2 / 2)
    dx_right = tf.minimum(boxes1_0 + boxes1_2 / 2, boxes2_0 + boxes2_2 / 2)

    dx_overlap = dx_right - dx_left
    dx_overlap = tf.maximum(dx_overlap, tf.zeros_like(dx_overlap))

    dy_left = tf.maximum(boxes1_1 - boxes1_3 / 2, boxes2_1 - boxes2_3 / 2)
    dy_right = tf.minimum(boxes1_1 + boxes1_3 / 2, boxes2_1 + boxes2_3 / 2)

    dy_overlap = dy_right - dy_left
    dy_overlap = tf.maximum(dy_overlap, tf.zeros_like(dy_overlap))

    intersections = dx_overlap * dy_overlap

    unions = boxes1_2 * boxes1_3 + boxes2_2 * boxes2_3 - intersections

    return intersections / tf.maximum(unions,1e-10)
    

def build_targets_masks(pred_boxes, ground_truth, batch_size, anchors_num, nH, nW, thresh, reduction):
    orgimg_w = 304
    orgimg_h = 224
    nAnchors = anchors_num*nH*nW
    object_no_detections = []
    batch_num = 32

    for b in range(batch_num):
        ground_truth_tensor = tf.reshape(ground_truth[b],[-1,5])
        groundtruth_num     = tf.shape(ground_truth_tensor)[0]

        # if tf.reduce_sum(ground_truth_tensor) == 0:   # No gt for this image
        #     print("tf.reduce_sum(ground_truth_tensor) == 0")
        #     continue
    
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors]

        scale = tf.constant([[orgimg_w,orgimg_h,orgimg_w,orgimg_h]],tf.float32)
        gt = ground_truth_tensor[:,1:5] * scale / tf.constant(reduction)

        # Set confidence mask of matching detections to 0
        iou_gt_pred = bbox_ious_tf(gt, cur_pred_boxes)

        iou_gt_pred = tf.reshape(iou_gt_pred,[1,groundtruth_num,anchors_num,nH*nW])

        mask = tf.cast((tf.reduce_sum(tf.cast((iou_gt_pred > thresh),tf.float32),1) >= 1),tf.float32)
        object_no_mask = tf.ones_like(mask, tf.float32) - mask

        if object_no_detections==[]:
            object_no_detections = object_no_mask
        else:
            object_no_detections = tf.concat([object_no_detections,object_no_mask],0)

    return object_no_detections


def yolo_loss(  target,
                output,
                coord_mask, object_detections, object_no_detections_gt_anch, gt_coord, gt_conf,
                object_scale=5.0,
                no_object_scale=1.0, coordinates_scale=3.0
            ):
    anchors = tf.constant([[16, 16], [48, 48], [80, 80], [112,112], [160,160]], tf.float32) / tf.constant(16.0)
    anchors_num = tf.shape(anchors)[0]
    output_shape = tf.shape(output) #(n,c,h,w)
    batch_size = output_shape[0]
    nH = output_shape[2]
    nW = output_shape[3]

    #reshape output as [batch,anchor,x-y-w-h-conf,w*h]
    output = tf.reshape(output,[batch_size, anchors_num, -1, nW*nH])
    
    coord_xy = tf.sigmoid(output[:, :, :2])  # x,y
    coord_wh = output[:, :, 2:4]  # w,h
    pre_coord = tf.concat([coord_xy,coord_wh], 2)
    pre_conf = tf.sigmoid(output[:, :, 4])

    # Create prediction boxes
    lin_x = tf.reshape(tf.tile(tf.reshape(tf.linspace(0.0, tf.cast(nW - 1,tf.float32), nW),[1,nW]),[nH,1]),[nW*nH])
    lin_y = tf.reshape(tf.tile(tf.reshape(tf.linspace(0.0, tf.cast(nH - 1,tf.float32), nH),[nH,1]),[1,nW]),[nW*nH])
    pred_boxes_x = tf.reshape(pre_coord[:, :, 0] + lin_x,[-1,1])
    pred_boxes_y = tf.reshape(pre_coord[:, :, 1] + lin_y,[-1,1])

    anchor_w = tf.reshape(anchors[:, 0],[anchors_num,1])
    anchor_h = tf.reshape(anchors[:, 1],[anchors_num,1])
    pred_boxes_w = tf.reshape(tf.exp(pre_coord[:, :, 2]) * anchor_w,[-1,1])
    pred_boxes_h = tf.reshape(tf.exp(pre_coord[:, :, 3]) * anchor_h,[-1,1])

    #x,y,w,h
    pred_boxes = tf.concat([pred_boxes_x,pred_boxes_y,pred_boxes_w,pred_boxes_h],1)

    object_no_detections_gt_pre = build_targets_masks(pred_boxes, target, batch_size, anchors_num, nH, nW, thresh = 0.6, reduction = 16.0)

    object_no_detections = object_no_detections_gt_pre * object_no_detections_gt_anch

    # coord
    coord_mask = tf.tile(coord_mask,[1,1,2,1])
    pre_coord_center, gt_coord_center = pre_coord[:, :, :2], gt_coord[:, :, :2]
    pre_coord_wh, gt_coord_wh = pre_coord[:, :, 2:], gt_coord[:, :, 2:]

    # Compute losses
    loss_coord_center = 2.0 * 1.0 * coordinates_scale * tf.reduce_sum(coord_mask * tf.keras.backend.binary_crossentropy(gt_coord_center, pre_coord_center))
    loss_coord_wh = 2.0 * 1.5 * coordinates_scale * tf.reduce_sum(coord_mask * tf.losses.huber_loss(gt_coord_wh, pre_coord_wh, reduction=tf.losses.Reduction.NONE))
    loss_coord = loss_coord_center + loss_coord_wh

    loss_conf_pos = 1.0 * object_scale * tf.reduce_sum(object_detections * tf.keras.backend.binary_crossentropy(gt_conf, pre_conf))
    loss_conf_neg = 1.0 * no_object_scale * tf.reduce_sum(object_no_detections * tf.keras.backend.binary_crossentropy(gt_conf, pre_conf))
    loss_conf = loss_conf_pos + loss_conf_neg

    loss_tot = (loss_coord + loss_conf) / tf.cast(batch_size,tf.float32)

    return loss_tot