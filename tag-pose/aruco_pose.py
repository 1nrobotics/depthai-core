#!/usr/bin/env python3

import argparse
import time

import cv2
import depthai as dai
import numpy as np


CAMERA_SOCKET = dai.CameraBoardSocket.CAM_B
TARGET_FPS = 30.0
DEFAULT_DICT = "DICT_4X4_50"


def build_argparser():
    parser = argparse.ArgumentParser(description="Detect ArUco markers with DepthAI v3 and estimate pose on the host.")
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.16,
        help="Physical ArUco marker edge length in meters. Default: 0.16",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default=DEFAULT_DICT,
        help=f"OpenCV ArUco dictionary name. Default: {DEFAULT_DICT}",
    )
    return parser


def get_camera_features(device: dai.Device, socket: dai.CameraBoardSocket):
    for features in device.getConnectedCameraFeatures():
        if features.socket == socket:
            return features
    raise RuntimeError(f"Camera socket {socket} not found on this device")


def select_highest_resolution_config(features, target_fps: float):
    valid_configs = [cfg for cfg in features.configs if cfg.maxFps >= target_fps]
    if not valid_configs:
        valid_configs = list(features.configs)
    if not valid_configs:
        raise RuntimeError(f"No sensor configs reported for {features.socket}")
    return max(valid_configs, key=lambda cfg: (cfg.width * cfg.height, cfg.maxFps))


def create_object_points(marker_size_m: float):
    half_size = marker_size_m / 2.0
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float32,
    )


def draw_pose_axes(frame, camera_matrix, distortion_coeffs, rvec, tvec, marker_size_m: float):
    axis_length = marker_size_m * 0.5
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, -axis_length],
        ],
        dtype=np.float32,
    )
    projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, distortion_coeffs)
    origin, x_axis, y_axis, z_axis = projected_points.reshape(-1, 2).astype(int)
    cv2.line(frame, tuple(origin), tuple(x_axis), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(frame, tuple(origin), tuple(y_axis), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(frame, tuple(origin), tuple(z_axis), (255, 0, 0), 2, cv2.LINE_AA)


def compute_reprojection(object_points, image_points, camera_matrix, distortion_coeffs, rvec, tvec):
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    per_corner_error = np.linalg.norm(image_points - projected_points, axis=1)
    rmse_px = float(np.sqrt(np.mean(per_corner_error ** 2)))
    return projected_points, rmse_px


def draw_reprojection(frame, image_points, projected_points):
    for observed, projected in zip(image_points.astype(int), projected_points.astype(int)):
        cv2.circle(frame, tuple(projected), 4, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.line(frame, tuple(observed), tuple(projected), (0, 255, 255), 1, cv2.LINE_AA)


def get_aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include cv2.aruco. Install opencv-contrib-python in your environment.")
    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def create_detector(dictionary):
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def get_supported_dictionaries():
    if not hasattr(cv2, "aruco"):
        return []
    return sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))


args = build_argparser().parse_args()
print(f"OpenCV version: {cv2.__version__}")
supported_dictionaries = get_supported_dictionaries()
print(f"Supported dictionaries ({len(supported_dictionaries)}): {', '.join(supported_dictionaries)}")
aruco_dictionary = get_aruco_dictionary(args.dictionary)
aruco_detector = create_detector(aruco_dictionary)

with dai.Pipeline() as pipeline:
    device = pipeline.getDefaultDevice()
    calibration = device.readCalibration()
    camera_features = get_camera_features(device, CAMERA_SOCKET)
    selected_config = select_highest_resolution_config(camera_features, TARGET_FPS)
    output_size = (selected_config.width, selected_config.height)
    output_fps = min(TARGET_FPS, selected_config.maxFps)

    print(
        f"Using {CAMERA_SOCKET} sensor={camera_features.sensorName} "
        f"resolution={output_size[0]}x{output_size[1]} fps={output_fps:g}"
    )

    host_camera = pipeline.create(dai.node.Camera).build(CAMERA_SOCKET, output_size, output_fps)
    output_queue = host_camera.requestOutput(output_size, dai.ImgFrame.Type.GRAY8, fps=output_fps).createOutputQueue()

    color = (0, 255, 0)
    start_time = time.monotonic()
    counter = 0
    fps = 0.0

    pipeline.start()
    while pipeline.isRunning():
        in_frame = output_queue.get()
        assert isinstance(in_frame, dai.ImgFrame)
        frame = in_frame.getCvFrame()
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        counter += 1
        current_time = time.monotonic()
        if (current_time - start_time) > 1:
            fps = counter / (current_time - start_time)
            counter = 0
            start_time = current_time

        camera_socket = dai.CameraBoardSocket(in_frame.getInstanceNum())
        camera_matrix = np.array(
            calibration.getCameraIntrinsics(camera_socket, in_frame.getWidth(), in_frame.getHeight()),
            dtype=np.float32,
        )
        distortion_coeffs = np.array(calibration.getDistortionCoefficients(camera_socket), dtype=np.float32)

        corners_list, ids, _ = aruco_detector.detectMarkers(frame)

        if ids is not None:
            object_points = create_object_points(args.marker_size)
            ids = ids.flatten()

            for marker_id, corners in zip(ids, corners_list):
                image_points = corners.reshape(4, 2).astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    distortion_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                contour = image_points.astype(int).reshape((-1, 1, 2))
                cv2.polylines(display, [contour], True, color, 2, cv2.LINE_AA)

                center = tuple(np.mean(image_points, axis=0).astype(int))
                if success:
                    projected_points, reprojection_rmse_px = compute_reprojection(
                        object_points, image_points, camera_matrix, distortion_coeffs, rvec, tvec
                    )
                    draw_pose_axes(display, camera_matrix, distortion_coeffs, rvec, tvec, args.marker_size)
                    draw_reprojection(display, image_points, projected_points)
                    distance_m = float(np.linalg.norm(tvec))
                    print(
                        f"id={int(marker_id)} tvec={tvec.ravel()} rvec={rvec.ravel()} "
                        f"distance_m={distance_m:.3f} reprojection_rmse_px={reprojection_rmse_px:.2f}"
                    )
                    pose_text = f"ID:{int(marker_id)} Z:{tvec[2][0]:.2f}m D:{distance_m:.2f}m E:{reprojection_rmse_px:.2f}px"
                else:
                    pose_text = f"ID:{int(marker_id)}"

                cv2.putText(display, pose_text, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.putText(display, f"fps: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color)
        cv2.imshow("aruco detections", display)
        if cv2.waitKey(1) == ord("q"):
            break
