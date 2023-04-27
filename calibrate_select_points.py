import os
import cv2
import numpy as np

from common import create_run_path, splitfn, show_img_with_kpts, calibrate, find_hom


class App:
    def __init__(self,
                 camera_img_path,
                 pattern_img_path,
                 camera_kpts_path=None,
                 pattern_kpts_path=None,
                 camera_select_kpts_path=None,
                 pattern_select_kpts_path=None,
                 fisheye=False,
                 random_select=-1):
        self.fisheye = fisheye
        self.random_select = random_select
        self.save_root = create_run_path()

        self.camera_window = 'camera'
        self.camera_img_path = camera_img_path
        self.camera_img = cv2.imread(camera_img_path)
        self.camera_kpts = None if camera_kpts_path is None else np.load(camera_kpts_path)['kpts']
        self.camera_kpts_select = self.read_select_kpts(camera_select_kpts_path)
        self.camera_show = None

        self.pattern_window = 'pattern'
        self.pattern_img_path = pattern_img_path
        self.pattern_img = cv2.imread(pattern_img_path)
        self.pattern_kpts = None if pattern_kpts_path is None else np.load(pattern_kpts_path)['kpts']
        self.pattern_kpts_select = self.read_select_kpts(pattern_select_kpts_path)
        self.pattern_show = None

    @staticmethod
    def auto_select_kpt(xy, kpts, dis=25):
        x, y = xy
        if kpts is None:
            return xy

        new_xy = kpts[np.argmin(np.sum((kpts - [x, y]) ** 2, 1))]
        if np.sum((new_xy - [x, y]) ** 2) < dis ** 2:
            return new_xy
        else:
            return xy

    @staticmethod
    def read_select_kpts(path):
        return [] if path is None else np.load(path)['kpts'].astype(np.int16).tolist()

    def check_select_kpts(self):
        camera_n = len(self.camera_kpts_select)
        pattern_n = len(self.pattern_kpts_select)
        print(f'相机取点: {camera_n}个, 模板图取点: {pattern_n}个')
        if camera_n == pattern_n and camera_n > 3:
            return True
        else:
            return False

    def camera_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = self.auto_select_kpt((x, y), self.camera_kpts)
            self.camera_kpts_select.append((int(x), int(y)))

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.camera_kpts_select):
                self.camera_kpts_select.pop(-1)

        self.camera_show = show_img_with_kpts(
            self.camera_img, self.camera_kpts_select, self.camera_window, kpt_size=5, font_size=1
        )

    def pattern_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = self.auto_select_kpt((x, y), self.pattern_kpts)
            self.pattern_kpts_select.append((int(x), int(y)))

        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.pattern_kpts_select):
                self.pattern_kpts_select.pop(-1)

        self.pattern_show = show_img_with_kpts(
            self.pattern_img, self.pattern_kpts_select, self.pattern_window, kpt_size=10, font_size=2
        )

    def calibrate(self):
        _, name, _ = splitfn(self.camera_img_path)
        cv2.imwrite(os.path.join(self.save_root, f"{name}_camera_select_kpts.jpg"), self.camera_show)
        cv2.imwrite(os.path.join(self.save_root, f"{name}_pattern_select_kpts.jpg"), self.pattern_show)

        camera_kpts = np.array(self.camera_kpts_select, dtype=np.float32)
        np.savez(os.path.join(self.save_root, f"{name}_camera_select_kpts.npz"), kpts=camera_kpts)
        pattern_kpts = np.array(self.pattern_kpts_select, dtype=np.float32)
        np.savez(os.path.join(self.save_root, f"{name}_pattern_select_kpts.npz"), kpts=pattern_kpts)
        pattern_kpts = np.concatenate((pattern_kpts, np.zeros((len(pattern_kpts), 1))), axis=1).astype(np.float32)

        camera_matrix, dist_coefs, _rvecs, _tvecs, newcameramtx, dst, dst_full = calibrate(
            self.camera_img, camera_kpts, pattern_kpts, self.fisheye
        )
        hom, dst_hom = find_hom(dst_full, self.pattern_img, pattern_kpts, _rvecs[0], _tvecs[0], newcameramtx)

        cv2.namedWindow("dst", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("dst_full", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("dst_hom", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("dst", dst)
        cv2.imshow("dst_full", dst_full)
        cv2.imshow("dst_hom", dst_hom)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(self.save_root, f"{name}_result.jpg"), dst)
        cv2.imwrite(os.path.join(self.save_root, f"{name}_result_full.jpg"), dst_full)
        cv2.imwrite(os.path.join(self.save_root, f"{name}_hom.jpg"), dst_hom)
        np.savez(os.path.join(self.save_root, f"{name}_calibrate_parm.npz"),
                 camera_matrix=camera_matrix,
                 dist_coefs=dist_coefs,
                 _rvecs=_rvecs,
                 _tvecs=_tvecs)
        np.savez(os.path.join(self.save_root, f"{name}_hom.npz"), hom=hom)

    def run(self):
        cv2.namedWindow(self.camera_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.camera_window, self.camera_mouse)

        cv2.namedWindow(self.pattern_window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(self.pattern_window, self.pattern_mouse)

        while True:
            self.camera_show = show_img_with_kpts(
                self.camera_img, self.camera_kpts_select, self.camera_window, kpt_size=5, font_size=1
            )
            self.pattern_show = show_img_with_kpts(
                self.pattern_img, self.pattern_kpts_select, self.pattern_window, kpt_size=10, font_size=2
            )
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                if self.check_select_kpts():
                    cv2.destroyAllWindows()
                    print("开始矫正")
                    self.calibrate()
                    break
                else:
                    print("相机与模板图取点数量不同或数量太少")


if __name__ == "__main__":
    calibration = App(
        camera_img_path='data/nike/left_side_2.jpg',
        pattern_img_path='data/pattern/court.jpg',
        camera_kpts_path=None,
        pattern_kpts_path='data/pattern/court_kpts.npz',
        camera_select_kpts_path='data/nike/left_side_2_camera_select_kpts.npz',
        pattern_select_kpts_path='data/nike/left_side_2_pattern_select_kpts.npz',
        fisheye=False,
        random_select=-1
    )
    calibration.run()
