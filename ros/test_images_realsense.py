#!/usr/bin/env python
"""Test hamer on ros images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn
import tf
import rosnode
import message_filters
import cv2
import threading
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import rospy
import ros_numpy
import copy
import scipy.io
import json
from typing import Dict, Optional
import matplotlib.pyplot as plt
from matplotlib import patches
from pathlib import Path

import std_msgs.msg
import sensor_msgs.msg
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

import open3d as o3d
from scipy.optimize import minimize
from sklearn.neighbors import KDTree

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

lock = threading.Lock()
lock_tf = threading.Lock()
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


def obj_funcion(x, vertices, translation, K1, K2, kd_tree):
    # projection 1
    V1 = vertices + translation
    x1 = K1 @ V1.T
    x1[0, :] /= x1[2, :]
    x1[1, :] /= x1[2, :]

    # projection 2
    V2 = vertices + x
    x2 = K2 @ V2.T
    x2[0, :] /= x2[2, :]
    x2[1, :] /= x2[2, :]

    # 3D distances
    distances, indices = kd_tree.query(V2)
    distances = distances.astype(np.float32).reshape(-1)
    error_3d = np.mean(distances)

    # error
    error_2d = np.square(x1[:2] - x2[:2]).mean()
    # print('error 2d', error_2d, 'error_3d', error_3d)
    return error_2d + 100 * error_3d


def create_marker(marker_id, position, marker_type=Marker.SPHERE, color=None):
    """
    Function to create a single marker
    """
    marker = Marker()
    marker.header.frame_id = "measured/camera_color_optical_frame"  # Coordinate frame (e.g., "base_link")
    marker.header.stamp = rospy.Time.now()
    marker.ns = "example_markers"
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD

    # Set position
    marker.pose.position = position
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Set scale (size of marker)
    marker.scale.x = 0.05  # For sphere radius or cube size
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    # Set color (default is white)
    if color is None:
        marker.color.r = 0.0  # Red
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque
    else:
        marker.color = color

    return marker


class ImageListener:

    def __init__(self, model, detector, cpm, renderer, model_cfg):

        self.model = model
        self.detector = detector
        self.cpm = cpm
        self.renderer = renderer
        self.model_cfg = model_cfg
        
        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.results = None
        self.counter = 0
        self.output_dir = 'output/real_world'

        # initialize a node
        rospy.init_node("seg_rgb")
        self.image_pub = rospy.Publisher('hamer_image', Image, queue_size=10)
        self.marker_pub = rospy.Publisher('hamer_marker_array', MarkerArray, queue_size=10)

        self.base_frame = 'measured/camera_color_optical_frame'
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
        msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)

        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsic_matrix = intrinsics
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)

        # start pose thread
        self.start_publishing_tf()


    def start_publishing_tf(self):
        self.stop_event = threading.Event()
        self.tf_thread = threading.Thread(target=self.tf_thread_func)
        self.tf_thread.start()


    def stop_publishing_tf(self):
        if self.tf_thread is None:
            return False
        self.stop_event.set()
        self.tf_thread.join()
        return True        


    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = ros_numpy.numpify(depth)
        elif depth.encoding == '16UC1':
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = ros_numpy.numpify(rgb)

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    # publish 3D location of objects
    def tf_thread_func(self):
        rate = rospy.Rate(10.)
        while not self.stop_event.is_set() and not rospy.is_shutdown():

            # publish 3D object location
            with lock_tf:
                results = self.results
                if results is None:
                    continue

                # backprojection
                depth_img = self.depth.copy()
                height = depth_img.shape[0]
                width = depth_img.shape[1]
                xyz_img = compute_xyz(depth_img, self.fx, self.fy, self.px, self.py, height, width)

                # compute 3D centroids
                # Create a MarkerArray
                marker_array = MarkerArray()                
                for i, result in enumerate(results):
                    mask = result > 0

                    xyz = xyz_img[mask]
                    selection = xyz[:, 2] > 0
                    xyz = xyz[selection]
                    center = np.mean(xyz, axis=0)

                    # Create a sphere marker
                    marker = create_marker(i, Point(center[0], center[1], center[2]), Marker.SPHERE)
                    marker_array.markers.append(marker)
                self.marker_pub.publish(marker_array)

            rate.sleep()
            # self.stop_event.wait(timeout=0.1)      


    def run_network(self):

        with lock:
            if listener.im is None:
              return
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        print('===========================================')

        # input imaeg: RGB numpy array.
        img_cv2 = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)

        # Detect humans in image
        det_out = self.detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            return

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_box_centers = []
        all_box_sizes = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                center = batch["box_center"][n].cpu().numpy()
                size = batch["box_size"][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_box_centers.append(center)
                all_box_sizes.append(size)

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            print('%d hands' % (len(all_verts)))
            cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            # backprojection
            depth_img = self.depth.copy()
            height = depth_img.shape[0]
            width = depth_img.shape[1]
            xyz_img = compute_xyz(depth_img, self.fx, self.fy, self.px, self.py, height, width)            

            # for each hand
            results = []
            for i in range(batch_size):
                cam_view = self.renderer.render_rgba_multiple([all_verts[i]], cam_t=[all_cam_t[i]], render_res=img_size[i], is_right=all_right, **misc_args)

                # get the hand points
                mask = cam_view[:,:,3] > 0
                # eroder mask
                kernel = np.ones((5, 5), np.uint8) 
                mask = cv2.erode(mask.astype(np.uint8), kernel)
                if np.sum(mask) == 0:
                    continue
                results.append(mask)

                # mask = 1 - mask
                # # convert depth to point cloud
                # depth_pc = DepthPointCloud(depth_img, self.intrinsic_matrix, camera_pose=np.eye(4), target_mask=mask, threshold=10.0)
                # print(depth_pc.points.shape)

                # solve new translation
                # print(mask.shape)
                # K = np.array([[12500, 0, 320], [0, 12500, 240], [0, 0, 1]]).astype(np.float32)
                # mask = mask > 0
                # xyz = xyz_img[mask]
                # print(xyz.shape)
                # selection = xyz[:, 2] > 0
                # xyz = xyz[selection]
                # x0 = np.mean(xyz, axis=0)
                # print(x0.shape)
                # kd_tree = KDTree(xyz)
                # res = minimize(obj_funcion, x0, method='nelder-mead',
                #             args=(all_verts[i], all_cam_t[i], K, self.intrinsic_matrix, kd_tree), options={'xatol': 1e-8, 'disp': True})
                # translation_new = res.x
                # print(res.x)
                # vertices = all_verts[i] + translation_new
                # center = np.mean(vertices, axis=0)
                # results.append(center)

                visualize = False
                if visualize:
                    fig = plt.figure()
                    ax = fig.add_subplot(2, 2, 1)
                    plt.imshow(cam_view)

                    # show box
                    center = all_box_centers[i]
                    size = all_box_sizes[i]
                    plt.plot(center[0], center[1], 'o')
                    print('size', size)

                    # add bounding boxes to the image
                    xmin = center[0] - size / 2 
                    ymin = center[1] - size / 2
                    box = patches.Rectangle(
                        (xmin, ymin), size, size, edgecolor="red", facecolor="none"
                    )
                    ax.add_patch(box)

                    ax = fig.add_subplot(2, 2, 2)
                    plt.imshow(input_img)
                    # verify projection 1
                    vertices = all_verts[i] + all_cam_t[i]
                    print(K, vertices)
                    print(vertices.shape)
                    x2d = K @ vertices.T
                    x2d[0, :] /= x2d[2, :]
                    x2d[1, :] /= x2d[2, :]
                    plt.plot(x2d[0, :], x2d[1, :])
                    plt.title('projection using hamer camera')

                    ax = fig.add_subplot(2, 2, 3)
                    plt.imshow(input_img)
                    # verify projection 2
                    vertices = all_verts[i] + translation_new
                    x2d = self.intrinsic_matrix @ vertices.T
                    x2d[0, :] /= x2d[2, :]
                    x2d[1, :] /= x2d[2, :]
                    plt.plot(x2d[0, :], x2d[1, :])              
                    plt.title('projection using realsense camera')

                    ax = fig.add_subplot(2, 2, 4, projection='3d')
                    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='o')
                    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], marker='o', color='r')

                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    plt.show()
            self.results = results

            # publish segmentation images
            input_img_overlay *= 255
            rgb_msg = ros_numpy.msgify(Image, input_img_overlay.astype(np.uint8), 'rgb8')
            rgb_msg.header.stamp = rgb_frame_stamp
            rgb_msg.header.frame_id = rgb_frame_id
            self.image_pub.publish(rgb_msg)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a hamer network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')                        
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    # image listener
    listener = ImageListener(model, detector, cpm, renderer, model_cfg)
    while not rospy.is_shutdown():
       listener.run_network()