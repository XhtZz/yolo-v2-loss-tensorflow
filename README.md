# yolo-v2-loss

#core function

def yolo_loss(  target,   
                output,   
                coord_mask, object_detections, object_no_detections_gt_anch, gt_coord, gt_conf,  
                object_scale=5.0,  
                no_object_scale=1.0,   
                coordinates_scale=3.0  
            )  
#parameters production              
target:   #label  
output:   #net output  
coord_mask, object_detections, object_no_detections_gt_anch, gt_coord, gt_conf：  
#groundtruth related parameters that obtained by function build_groundTrue_targets_masks in file postproc.py  
object_scale：  
no_object_scale:   
coordinates_scale:        
