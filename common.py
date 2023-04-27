import os
import cv2
import numpy as np

white = (255, 255, 255)
black = (0, 0, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def create_run_path():
    root = 'runs'
    check_path(root)
    n = 1
    while True:
        path = os.path.join(root, f'run{n}')
        if os.path.exists(path):
            n += 1
        else:
            os.mkdir(path)
            break
    return path


def kpts_hom_tran(kpts, hom):
    """
    kpts: nx3 原始点坐标
    hom: 3x3 单应性矩阵
    return: nx3 变换后坐标
    """
    kpts_tran = np.matmul(hom, kpts.T).T
    return kpts_tran / kpts_tran[:, 2:]


def show_img_with_kpts(img, kpts, window, kpt_size=10, font_size=1):
    i_show = img.copy()
    for i, kpt in enumerate(kpts):
        x, y = kpt
        cv2.circle(i_show, (x, y), kpt_size, green, -1, 16)
        cv2.putText(i_show, str(i), (kpt[0] + 10, kpt[1]), cv2.FONT_HERSHEY_TRIPLEX, font_size, green, 1, cv2.LINE_AA)
    cv2.imshow(window, i_show)
    return i_show


def calibrate(camera_img, camera_kpts, pattern_kpts, fisheye):
    h, w, _ = camera_img.shape

    if fisheye:
        flags = 0
        flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        flags |= cv2.fisheye.CALIB_CHECK_COND
        flags |= cv2.fisheye.CALIB_FIX_SKEW
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        # retval, K, D, rvecs, tvecs
        rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.fisheye.calibrate(
            [pattern_kpts.reshape((-1, 1, 3))], [camera_kpts.reshape((-1, 1, 2))], (w, h), None, None,
            None, None, flags=flags, criteria=criteria
        )

        newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coefs, (w, h), None, None, 1, (w, h)
        )
        dst = cv2.fisheye.undistortImage(
            camera_img, camera_matrix, dist_coefs, None, newcameramtx, new_size=(w, h)
        )

        newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coefs, (w, h), None, None, 0, (w, h)
        )
        dst_full = cv2.fisheye.undistortImage(
            camera_img, camera_matrix, dist_coefs, None, newcameramtx, new_size=(w, h)
        )
    else:
        rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
            [pattern_kpts], [camera_kpts], (w, h), None, None
        )
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        dst = cv2.undistort(camera_img, camera_matrix, dist_coefs, None, newcameramtx)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 0, (w, h))
        dst_full = cv2.undistort(camera_img, camera_matrix, dist_coefs, None, newcameramtx)

    print(f'rms: {rms}')
    print(f'dist_coefs: {dist_coefs}')
    print(f'_rvecs: {_rvecs}')
    print(f'_tvecs: {_tvecs}')
    return camera_matrix, dist_coefs, _rvecs, _tvecs, newcameramtx, dst, dst_full


def find_hom(camera_img_undistort, pattern_img, pattern_kpts, _rvecs, _tvecs, newcameramtx):
    # 模板图点转为矫正后图像上的点
    undistort_kpts, _ = cv2.projectPoints(pattern_kpts, _rvecs, _tvecs, newcameramtx, None)
    undistort_kpts = undistort_kpts.reshape(-1, 2)

    hom, status = cv2.findHomography(undistort_kpts, pattern_kpts, cv2.RANSAC, 5)
    h, w = pattern_img.shape[:2]
    dst_hom = cv2.warpPerspective(camera_img_undistort, hom, (w, h))
    dst_hom = cv2.addWeighted(dst_hom, 0.7, pattern_img, 0.3, 0)
    return hom, dst_hom

