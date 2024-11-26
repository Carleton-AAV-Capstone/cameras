import os
import numpy as np
import cv2

CHESSBOARD_DIM = (9, 6)
SQUARE_SIZE = 25.0

mtx = np.zeros((3, 3), dtype=np.float64)
dist = np.zeros((5, 1), dtype=np.float64)

objp = np.zeros((CHESSBOARD_DIM[0] * CHESSBOARD_DIM[1], 3), np.float32)
objp[:, :2] = np.indices(CHESSBOARD_DIM).T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

FILL_RATIO_THRESHOLD = 0.6 # 60%

def load_calibration_data():
    calibration_files_exist = os.path.exists('camera_matrix.npy') and os.path.exists('distortion_coefficients.npy')

    if calibration_files_exist:
        mtx = np.load('camera_matrix.npy')
        dist = np.load('distortion_coefficients.npy')
        print("Loaded existing calibration data.")
        return mtx, dist
    else:
        print("Calibration data not found, starting calibration process...")
        mtx, dist = calibrate_camera()
        return mtx, dist


def calibrate_camera():
    """Perform camera calibration using chessboard images."""
    global objpoints, imgpoints

    capture = cv2.VideoCapture(0)  # Adjust the camera index if needed

    while len(objpoints) < 10:  # Minimum 10 frames for calibration can be changed
        ret, frame = capture.read()
        if not ret:
            print("Failed to capture image")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_DIM, None)

        if ret:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

            x, y, w, h = cv2.boundingRect(corners)

            # Calculate the fill ratio of the chessboard with respect to the image size
            img_height, img_width = gray.shape
            fill_ratio_width = w / img_width
            fill_ratio_height = h / img_height

            if fill_ratio_width >= FILL_RATIO_THRESHOLD and fill_ratio_height >= FILL_RATIO_THRESHOLD:
                imgpoints.append(corners)
                objpoints.append(objp)

                print(f"Chessboard accepted. Fill ratio: {fill_ratio_width:.2f} (width), {fill_ratio_height:.2f} (height)")

            else:
                print(f"Chessboard rejected. Fill ratio too small: {fill_ratio_width:.2f} (width), {fill_ratio_height:.2f} (height)")

            cv2.drawChessboardCorners(frame, CHESSBOARD_DIM, corners, ret)

        cv2.imshow("Chessboard Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop when 'q' is pressed
            break

    capture.release()
    cv2.destroyAllWindows()

    print("Performing camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Camera calibration successful.")
        # Save the calibration data
        np.save('camera_matrix.npy', mtx)
        np.save('distortion_coefficients.npy', dist)
        print("Calibration data saved.")
        return mtx, dist
    else:
        print("Calibration failed.")
        return None, None


def main():
    mtx, dist = load_calibration_data()

    capture = cv2.VideoCapture(0)  # Change the camera index for a different camera

    if not capture.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to capture image")
            break

        undistorted_frame = cv2.undistort(frame, mtx, dist)

        h, w = undistorted_frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        x, y, w, h = roi
        cropped_frame = undistorted_frame[y:y+h, x:x+w]

        cv2.imshow("Undistorted and Cropped Camera Feed", cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
