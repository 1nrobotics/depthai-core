git clone https://github.com/luxonis/depthai-core.git && cd depthai-core
python3 -m venv venv
source venv/bin/activate
# Installs library and requirements
python3 examples/python/install_requirements.py


cd examples/python
# Run YoloV6 detection example
python3 DetectionNetwork/detection_network.py
# Display all camera streams
python3 Camera/camera_all.py

# Run detector
python3 aruco-test/aruco_pose.py --marker-size 0.16 --dictionary DICT_6X6_250
apriltag online gen: https://chaitanyantr.github.io/apriltag.html
aruco online gen: https://chev.me/arucogen/
