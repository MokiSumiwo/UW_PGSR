#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import sys
import os
import json
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, depth_params=None):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    invdepthmap = None
    
    # --- DEBUG START ---
    if id == 0:
        print(f"\n[DEBUG loadCam] source_path: {getattr(args, 'source_path', 'MISSING')}")
        # Check for 'depths' (from command line) or '_depths' (internal default)
        depths_attr = getattr(args, 'depths', getattr(args, '_depths', 'MISSING'))
        print(f"[DEBUG loadCam] depths attribute: {depths_attr}")
        print(f"[DEBUG loadCam] image_name: {cam_info.image_name}")
        
        if hasattr(args, 'source_path') and depths_attr != 'MISSING':
            expected_path = os.path.join(args.source_path, depths_attr, cam_info.image_name + ".png")
            print(f"[DEBUG loadCam] Expected depth path: {expected_path}")
            print(f"[DEBUG loadCam] Exists: {os.path.exists(expected_path)}")
    # --- DEBUG END ---

    dro_pruning_active = getattr(args, 'dro_guided_pruning', False)
    depths_folder = getattr(args, 'depths', getattr(args, '_depths', None))
    if dro_pruning_active and hasattr(args, 'source_path') and depths_folder:
        depth_path = os.path.join(args.source_path, depths_folder, cam_info.image_name + ".png")
        if os.path.exists(depth_path):
            try:
                invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
            except Exception as e:
                print(f"Warning: Failed to load depth map {depth_path}: {e}")

    # --- Optional: fake air (pseudo clear-scene) image path ---
    # Priority:
    #  1) args.fake_air_source_path (explicit folder)
    #  2) auto-detect under <source_path>/<images_root>/<fake_air_folder_name>/
    fake_air_path = None
    try:
        fake_air_root = getattr(args, "fake_air_source_path", "") or ""
        fake_air_folder_name = getattr(args, "fake_air_folder_name", "fake_air_Deep_Sea-NN")
        if fake_air_root and os.path.isdir(fake_air_root):
            fake_air_dir = fake_air_root
        else:
            # Derive images_root from the training image path to stay compatible with custom args.images
            # e.g. .../<source_path>/images/ColorImage/xxx.png -> images_root=.../<source_path>/images
            images_dir = os.path.dirname(cam_info.image_path)
            images_root = os.path.dirname(images_dir)
            fake_air_dir = os.path.join(images_root, fake_air_folder_name)

        if os.path.isdir(fake_air_dir):
            # Prefer exact same filename (including extension) to match user promise "一一匹配"
            base = os.path.basename(cam_info.image_path)
            cand = os.path.join(fake_air_dir, base)
            if os.path.exists(cand):
                fake_air_path = cand
            else:
                # Fallback: match by stem with common extensions
                stem = cam_info.image_name
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    cand2 = os.path.join(fake_air_dir, stem + ext)
                    if os.path.exists(cand2):
                        fake_air_path = cand2
                        break
    except Exception as e:
        # Never break training because of optional pseudo supervision
        if id == 0:
            print(f"[Warning] Failed to resolve fake-air path for {cam_info.image_name}: {e}")

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=cam_info.global_id, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  invdepthmap=invdepthmap, depth_params=depth_params,
                  fake_air_path=fake_air_path)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    depth_params_dict = {}
    if hasattr(args, 'source_path'):
        candidates = [
            os.path.join(args.source_path, "sparse", "0", "depth_params.json"),
            os.path.join(args.source_path, "sparse", "depth_params.json")
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    with open(p, 'r') as f:
                        depth_params_dict = json.load(f)
                    print(f"Loaded depth params from {p}")
                    break
                except Exception as e:
                    print(f"Warning: Failed to load depth params from {p}: {e}")

    for id, c in enumerate(cam_infos):
        d_params = depth_params_dict.get(c.image_name, None)
        camera_list.append(loadCam(args, id, c, resolution_scale, d_params))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
