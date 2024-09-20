import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import argparse
import depthai as dai

ispScale = (1,4) # use (1,2) for higher resolution
camRes = dai.ColorCameraProperties.SensorResolution.THE_720_P
camSocket = dai.CameraBoardSocket.CAM_A

cam_options = ['rgb', 'cam', 'undistort']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input",
                    help="select camera input source for inference", default='cam', choices=cam_options)
parser.add_argument("-conf", "--confidence_thresh",
                    help="set the confidence threshold", default=0.3, type=float)
"""
parser.add_argument("-iou", "--iou_thresh",
                    help="set the NMS IoU threshold", default=0.4, type=float)
"""

def main():
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    args = parser.parse_args()
    cam_source = args.cam_input
    min_confidence = args.confidence_thresh

    model = YOLO("best.pt")

    if cam_source == "cam":
        for result in model.track(source=0, show=True, stream=True, agnostic_nms=True):
            visualizeDetections(result, model, box_annotator, min_confidence)

            if (cv2.waitKey(30) == 27):
                break
    elif cam_source == "rgb":
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(416, 416)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam.setFps(30)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        # # link RGB camera to XLinkOut node
        cam.video.link(xout_rgb.input)

        # Connect to the device and start the pipeline
        with dai.Device(pipeline) as device:
            # Get the output queue for RGB frames
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            # Retrieve calibration data for RGB Camera
            calibData = device.readCalibration()

            # Intrinsic parameters for RGB Camera
            intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)

            # Distortion coefficients for the RGB Camera
            distortion = calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)

            # Create intrinsic matrix and distortion coefficient matrix
            K = np.array(intrinsics)
            D = np.array(distortion)

            while True:
                # Get an RGB frame from the camera
                rgb_frame = q_rgb.get()
                frame = rgb_frame.getCvFrame()  # Convert to OpenCV format

                # Get frame dimensions
                h, w = frame.shape[:2]

                # Undistort image
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                frame = cv2.undistort(frame, K, D, None, new_camera_matrix)

                # crop image
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

                # Use YOLOv8 model to detect objects in the frame
                result = model(frame)[0]
                visualizeDetections(result, model, box_annotator, min_confidence)

                if (cv2.waitKey(30) == 27):
                    break
        
        # Release resources
        cv2.destroyAllWindows()
    elif cam_source == "undistort":
        with dai.Device() as device:
            calibData = device.readCalibration()
            pipeline = create_pipeline(calibData)
            device.startPipeline(pipeline)
    
            q = device.getOutputQueue(name="Undistorted", maxSize=4, blocking=False)
            
            while True:
                results = model.track(source=q.get().getCvFrame(), stream=True, show=True, agnostic_nms=True)
                
                for result in results:
                    visualizeDetections(result, model, box_annotator, min_confidence)

                if cv2.waitKey(1) == ord('q'):
                        break


def create_pipeline(calibData):
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setIspScale(ispScale)
    cam.setBoardSocket(camSocket)
    cam.setResolution(camRes)
    cam.setFps(20)

    manip = pipeline.create(dai.node.ImageManip)
    mesh, meshWidth, meshHeight = getMesh(calibData, cam.getIspSize())
    manip.setWarpMesh(mesh, meshWidth, meshHeight)
    manip.setMaxOutputFrameSize(cam.getIspWidth() * cam.getIspHeight() * 3 // 2)
    cam.isp.link(manip.inputImage)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("Undistorted")
    manip.out.link(cam_xout.input)

    # dist_xout = pipeline.create(dai.node.XLinkOut)
    # dist_xout.setStreamName("Distorted")
    # cam.isp.link(dist_xout.input)

    return pipeline


def getMesh(calibData, ispSize):
    M1 = np.array(calibData.getCameraIntrinsics(camSocket, ispSize[0], ispSize[1]))
    d1 = np.array(calibData.getDistortionCoefficients(camSocket))
    R1 = np.identity(3)
    mapX, mapY = cv2.initUndistortRectifyMap(M1, d1, R1, M1, ispSize, cv2.CV_32FC1)

    meshCellSize = 128 # Mesh size, if increased then reduces complexity
    mesh0 = []
    # Creates subsampled mesh which will be loaded on to device to undistort the image
    for y in range(mapX.shape[0] + 1): # iterating over height of the image
        if y % meshCellSize == 0:
            rowLeft = []
            for x in range(mapX.shape[1]): # iterating over width of the image
                if x % meshCellSize == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        rowLeft.append(mapX[y - 1, x - 1])
                        rowLeft.append(mapY[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        rowLeft.append(mapX[y - 1, x])
                        rowLeft.append(mapY[y - 1, x])
                    elif x == mapX.shape[1]:
                        rowLeft.append(mapX[y, x - 1])
                        rowLeft.append(mapY[y, x - 1])
                    else:
                        rowLeft.append(mapX[y, x])
                        rowLeft.append(mapY[y, x])
            if (mapX.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)

            mesh0.append(rowLeft)

    mesh0 = np.array(mesh0)
    meshWidth = mesh0.shape[1] // 2
    meshHeight = mesh0.shape[0]
    mesh0.resize(meshWidth * meshHeight, 2)

    mesh = list(map(tuple, mesh0))

    return mesh, meshWidth, meshHeight


def visualizeDetections(result, model, box_annotator, min_confidence):
    frame = result.orig_img
    detections = sv.Detections.from_yolov8(result)
    
    print(detections)

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    detections = detections[(detections.class_id != 60) & (detections.class_id != 0) & (detections.confidence >= min_confidence)]

    if len(detections) == 0:
        # Skip the current frame if no detections are found (Optimisation)
        return

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]

    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections,
        labels=labels
    )

    cv2.imshow("yolov8", frame)


if __name__ == "__main__":
    main()
