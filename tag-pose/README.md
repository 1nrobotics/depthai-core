# Tag Pose Estimation with DepthAI

This folder contains two small host-side pose-estimation demos:

- `aruco_pose.py`: detects ArUco markers with OpenCV and estimates pose with `solvePnP`
- `apriltag_pose.py`: detects AprilTags with the DepthAI AprilTag node and estimates pose with `solvePnP`

Both scripts follow the same idea:

1. Capture grayscale frames from one on-device camera
2. Detect the 4 marker corners in image pixels
3. Read the factory calibration for that camera from the device
4. Solve the 3D pose of the square marker relative to the camera
5. Draw the marker outline, pose axes, and print `tvec` / `rvec`

For `apriltag_pose.py`, you can choose where detection runs:

- `--detector-runtime device`: AprilTag detection runs on the OAK device
- `--detector-runtime host`: AprilTag detection runs on the host machine

In both cases, pose estimation still runs on the host with OpenCV.

## What The Pose Means

The scripts define the marker coordinate system at the center of the printed square:

- `+X`: to the marker's right
- `+Y`: to the marker's top edge
- `Z = 0`: the marker plane

The 3D corner model used by both scripts is:

```text
(-s/2, +s/2, 0)   (+s/2, +s/2, 0)
(-s/2, -s/2, 0)   (+s/2, -s/2, 0)
```

where `s` is the physical marker edge length in meters.

`cv2.solvePnP(..., flags=cv2.SOLVEPNP_IPPE_SQUARE)` returns:

- `rvec`: marker orientation relative to the camera
- `tvec`: marker position relative to the camera

More precisely, OpenCV gives the transform from marker coordinates into camera coordinates:

```text
X_camera = R * X_marker + t
```

where:

- `R = Rodrigues(rvec)`
- `t = tvec`

This means:

- `tvec[0]`: left/right offset of the marker center in the camera frame
- `tvec[1]`: up/down offset of the marker center in the camera frame
- `tvec[2]`: forward distance from the camera to the marker plane along the optical axis

The overlay also prints:

- `Z`: `tvec[2]`, the forward distance
- `D`: `norm(tvec)`, the straight-line distance from camera center to marker center

If you want the camera pose in the marker frame instead, invert the transform:

```python
R, _ = cv2.Rodrigues(rvec)
camera_position_in_marker_frame = -R.T @ tvec
camera_rotation_in_marker_frame = R.T
```

## Why `IPPE_SQUARE` Is Used

These markers are flat squares, so `cv2.SOLVEPNP_IPPE_SQUARE` is a good fit. It is designed for planar square targets and is usually more stable than a generic PnP solver for this exact problem.

## Axis Drawing Convention

The scripts draw:

- red: `+X`
- green: `+Y`
- blue: `-Z`

The blue line is drawn along `-Z` on purpose so the axis appears to come out toward the camera in the displayed image. The solved pose itself still comes from the standard `rvec` / `tvec` returned by OpenCV.

## Camera Selection

Both scripts currently use:

```python
CAMERA_SOCKET = dai.CameraBoardSocket.CAM_B
```

That means the demo runs on the camera connected to `CAM_B` in the device calibration. On `OAK-D` and `OAK-D-W`, the exact physical lens mapped to `CAM_B` can vary by hardware generation and calibration layout, so the safest rule is:

- plug in the device
- start the script
- read the startup line such as `Using CAM_B sensor=... resolution=...`

If you want to force another camera, edit `CAMERA_SOCKET` in the script.

## Before You Run

Clone the repo, create a virtual environment, and install the example requirements:

```bash
git clone https://github.com/luxonis/depthai-core.git
cd depthai-core
python3 -m venv venv
source venv/bin/activate
python3 examples/python/install_requirements.py
```

For ArUco, your OpenCV must include the `cv2.aruco` module. If you see an error about missing `cv2.aruco`, install an OpenCV build with contrib modules in your environment.

## Marker Preparation

Pick one marker family and print a marker at a known real size.

For the AprilTag demo in this folder, use the `36h11` family.

AprilTag generator:

- https://chaitanyantr.github.io/apriltag.html
- select the `36h11` family when generating the tag

ArUco generator:

- https://chev.me/arucogen/

Important:

- measure only the black square edge length
- pass that size in meters
- a typical small tag is about `2 cm`, so use `0.02`
- example: a `5 cm` tag uses `0.05`

Pose scale is only as good as this number. If the physical size is wrong, distance will be wrong.

For close-range tabletop testing, a `2 cm` tag is common. For easier first-time bring-up, especially with `OAK-D-W`, a larger tag such as `4 cm` to `8 cm` is usually easier to detect and gives more stable corners at moderate distance.

## Running On OAK-D

The steps are the same for `OAK-D` and `OAK-D-W`.

1. Connect the camera over USB.
2. Activate the virtual environment:

```bash
cd /path/to/depthai-core
source venv/bin/activate
```

3. Run one of the demos from the repository root.

AprilTag:

```bash
# use a printed AprilTag from the 36h11 family
python3 tag-pose/apriltag_pose.py --tag-size 0.02 --family TAG_36H11 --detector-runtime device
```

ArUco:

```bash
python3 tag-pose/aruco_pose.py --marker-size 0.02 --dictionary DICT_4X4_50
```

4. Hold the printed tag in front of the camera used by `CAM_B`.
5. If you are using a `2 cm` tag, start close to the camera, around `10 cm` to `40 cm`.
6. Move the tag slowly while keeping it mostly flat and fully visible.
7. Press `q` to quit.

What you should see:

- a green outline around the detected marker
- colored pose axes at the marker center
- magenta dots showing the reprojected marker corners from the estimated pose
- yellow lines from each detected corner to its reprojected corner
- `ID`, `Z`, `D`, and `E` drawn on the image
- console output like:

```text
id=0 tvec=[...] rvec=[...] distance_m=0.734 reprojection_rmse_px=0.82
```

Here `E` is the reprojection RMSE in pixels. Lower is better.

## Running On OAK-D-W

`OAK-D-W` works the same way, but the wider field of view changes practical behavior:

- the tag appears smaller at the same distance
- corner localization may get noisier near image edges
- a `2 cm` tag usually needs to be closer to the camera
- a larger printed tag is often easier for first validation

Recommended workflow for `OAK-D-W`:

1. If your tag is `2 cm`, begin very close, roughly `10 cm` to `30 cm` from the camera.
2. For easier bring-up, use a larger tag such as `4 cm` to `8 cm`.
3. Keep the tag near the center of the image while validating pose.
4. Once detection is stable, test wider-angle placements and longer distances.

Commands are the same:

```bash
# use a printed 36h11 AprilTag here
python3 tag-pose/apriltag_pose.py --tag-size 0.02 --family TAG_36H11 --detector-runtime device
python3 tag-pose/aruco_pose.py --marker-size 0.02 --dictionary DICT_4X4_50
```

If you want AprilTag detection to run on the host instead, use:

```bash
python3 tag-pose/apriltag_pose.py --tag-size 0.02 --family TAG_36H11 --detector-runtime host
```

## ArUco Example Dictionaries

You can pick any dictionary supported by your OpenCV build. A few common choices:

- `DICT_4X4_50`
- `DICT_5X5_100`
- `DICT_6X6_250`
- `DICT_APRILTAG_36h11` if your OpenCV build exposes it through `cv2.aruco`

Example:

```bash
python3 tag-pose/aruco_pose.py --marker-size 0.02 --dictionary DICT_6X6_250
```

## Practical Accuracy Tips

- Use a rigid, flat printout
- Enter the true marker size in meters
- A `2 cm` tag works best at short range; use a larger tag if you need more distance
- Keep the full marker inside the frame
- Avoid motion blur
- Avoid extreme tilt while validating first results
- Keep the tag away from strong glare
- Prefer the center of the image for the cleanest first test

## Verifying During Testing

Use the reprojection overlay as a quick confidence check:

- if the magenta reprojected corners sit almost on top of the detected corners, the pose is internally consistent
- short yellow lines mean low reprojection error
- long yellow lines mean the estimated pose is not matching the observed corners well

Practical guidance:

- under good conditions, errors around `1 px` or a few pixels are usually a good sign
- if `E` grows noticeably while the tag is still and clearly visible, check blur, lighting, tag size, and whether the tag is too close to the image edge
- compare `E` while changing distance and tilt; it should usually get worse as the tag gets smaller, blurrier, or more oblique

## Troubleshooting

`ModuleNotFoundError: No module named 'cv2'`

- install the example requirements inside the active virtual environment

`This OpenCV build does not include cv2.aruco`

- install an OpenCV build with contrib modules

No detections

- confirm the printed marker family matches the detector
- confirm the ArUco dictionary matches the printed marker
- increase marker size
- move closer
- improve lighting

Pose looks unstable

- verify the marker size value
- keep the marker flat
- reduce blur
- keep the marker near the image center

Wrong camera is being used

- change `CAMERA_SOCKET` in the script
- rerun and confirm the startup line printed by the script

## Code Notes

AprilTag pipeline:

- detection can run either on the device or on the host with `--detector-runtime`
- pose is still solved on the host with OpenCV using the returned corners

ArUco pipeline:

- detection and pose are both host-side
- OpenCV's `ArucoDetector` finds corners
- OpenCV `solvePnP` estimates pose

Both scripts use the device's stored intrinsics and distortion coefficients for the selected output size, which is what makes the metric pose estimate possible.
