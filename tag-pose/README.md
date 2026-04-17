git clone https://github.com/luxonis/depthai-core.git && cd depthai-core
python3 -m venv venv
source venv/bin/activate
# Installs library and requirements
python3 examples/python/install_requirements.py
# Required for ArUco (`cv2.aruco`)
pip install -U opencv-contrib-python

cd tag-pose

# Run detectors
python3 aruco_pose.py --marker-size 0.16 --dictionary DICT_6X6_250
python3 apriltag_pose.py --tag-size 0.16
apriltag online gen: https://chaitanyantr.github.io/apriltag.html
aruco online gen: https://chev.me/arucogen/
