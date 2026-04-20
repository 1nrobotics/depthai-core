#!/usr/bin/env python3

import argparse
import time

import cv2
import depthai as dai
import numpy as np


CAMERA_SOCKET = dai.CameraBoardSocket.CAM_B
TARGET_FPS = 30.0


def build_argparser():
    parser = argparse.ArgumentParser(description="Detect AprilTags with DepthAI v3 and estimate pose on the host.")
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.16,
        help="Physical AprilTag edge length in meters. Default: 0.16",
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


def to_int(point: dai.Point2f):
    return (int(point.x), int(point.y))


def create_object_points(tag_size_m: float):
    half_size = tag_size_m / 2.0
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_pose(tag, camera_matrix, distortion_coeffs, tag_size_m: float):
    image_points = np.array(
        [
            [tag.topLeft.x, tag.topLeft.y],
            [tag.topRight.x, tag.topRight.y],
            [tag.bottomRight.x, tag.bottomRight.y],
            [tag.bottomLeft.x, tag.bottomLeft.y],
        ],
        dtype=np.float32,
    )
    success, rvec, tvec = cv2.solvePnP(
        create_object_points(tag_size_m),
        image_points,
        camera_matrix,
        distortion_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return success, rvec, tvec


def draw_pose_axes(frame, camera_matrix, distortion_coeffs, rvec, tvec, tag_size_m: float):
    axis_length = tag_size_m * 0.5
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


args = build_argparser().parse_args()

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

    hostCamera = pipeline.create(dai.node.Camera).build(CAMERA_SOCKET, output_size, output_fps)
    aprilTagNode = pipeline.create(dai.node.AprilTag)
    hostCamera.requestOutput(output_size, dai.ImgFrame.Type.GRAY8, fps=output_fps).link(aprilTagNode.inputImage)
    passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
    outQueue = aprilTagNode.out.createOutputQueue()

    color = (0, 255, 0)
    start_time = time.monotonic()
    counter = 0
    fps = 0.0
    camera_matrix = None
    distortion_coeffs = None

    pipeline.start()
    while pipeline.isRunning():
        aprilTagMessage = outQueue.get()
        assert(isinstance(aprilTagMessage, dai.AprilTags))
        aprilTags = aprilTagMessage.aprilTags

        counter += 1
        current_time = time.monotonic()
        if (current_time - start_time) > 1:
            fps = counter / (current_time - start_time)
            counter = 0
            start_time = current_time

        passthroughImage: dai.ImgFrame = passthroughOutputQueue.get()
        frame = passthroughImage.getCvFrame()

        if camera_matrix is None or distortion_coeffs is None:
            camera_socket = dai.CameraBoardSocket(passthroughImage.getInstanceNum())
            camera_matrix = np.array(
                calibration.getCameraIntrinsics(camera_socket, passthroughImage.getWidth(), passthroughImage.getHeight()),
                dtype=np.float32,
            )
            distortion_coeffs = np.array(calibration.getDistortionCoefficients(camera_socket), dtype=np.float32)

        for tag in aprilTags:
            topLeft = to_int(tag.topLeft)
            topRight = to_int(tag.topRight)
            bottomRight = to_int(tag.bottomRight)
            bottomLeft = to_int(tag.bottomLeft)

            center = (int((topLeft[0] + bottomRight[0]) / 2), int((topLeft[1] + bottomRight[1]) / 2))

            cv2.line(frame, topLeft, topRight, color, 2, cv2.LINE_AA)
            cv2.line(frame, topRight, bottomRight, color, 2, cv2.LINE_AA)
            cv2.line(frame, bottomRight, bottomLeft, color, 2, cv2.LINE_AA)
            cv2.line(frame, bottomLeft, topLeft, color, 2, cv2.LINE_AA)

            success, rvec, tvec = estimate_pose(tag, camera_matrix, distortion_coeffs, args.tag_size)
            if success:
                draw_pose_axes(frame, camera_matrix, distortion_coeffs, rvec, tvec, args.tag_size)
                distance_m = float(np.linalg.norm(tvec))
                print(f"id={tag.id} tvec={tvec.ravel()} rvec={rvec.ravel()} distance_m={distance_m:.3f}")
                pose_text = f"ID:{tag.id} Z:{tvec[2][0]:.2f}m D:{distance_m:.2f}m"
            else:
                pose_text = f"ID:{tag.id}"

            cv2.putText(frame, pose_text, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.putText(frame, f"fps: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color)
        cv2.imshow("detections", frame)
        if cv2.waitKey(1) == ord("q"):
            break
