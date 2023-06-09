import os
import cv2
import numpy as np
import argparse

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
        self.camera_kpts_select_raw = None
        self.camera_show = None

        self.pattern_window = 'pattern'
        self.pattern_img_path = pattern_img_path
        self.pattern_img = cv2.imread(pattern_img_path)
        self.pattern_kpts = None if pattern_kpts_path is None else np.load(pattern_kpts_path)['kpts']
        self.pattern_kpts_select = self.read_select_kpts(pattern_select_kpts_path)
        self.pattern_kpts_select_raw = None
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
        if camera_n == pattern_n and camera_n >= 5:
            return True
        else:
            return False

    def random_select_kpts(self):
        n = len(self.camera_kpts_select_raw)
        if self.random_select < 5:
            print('随机取点数小于5, 不进入随机取点')
            self.random_select = -1
        else:
            camera_rand_kpts = []
            pattern_rand_kpts = []
            rand_idx = np.arange(len(self.camera_kpts_select_raw))
            np.random.shuffle(rand_idx)
            for i in rand_idx[:self.random_select]:
                camera_rand_kpts.append(self.camera_kpts_select_raw[i])
                pattern_rand_kpts.append(self.pattern_kpts_select_raw[i])
            self.camera_kpts_select = camera_rand_kpts
            self.pattern_kpts_select = pattern_rand_kpts
            self.camera_show = show_img_with_kpts(
                self.camera_img, self.camera_kpts_select, self.camera_window, kpt_size=5, font_size=1
            )
            self.pattern_show = show_img_with_kpts(
                self.pattern_img, self.pattern_kpts_select, self.pattern_window, kpt_size=10, font_size=2
            )
            print(f'在手动选择的{n}个点中随机选取{self.random_select}个点')
            self.calibrate()

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

        camera_kpts = np.array(self.camera_kpts_select, dtype=np.float32)
        pattern_kpts2 = np.array(self.pattern_kpts_select, dtype=np.float32)
        pattern_kpts = np.concatenate((pattern_kpts2, np.zeros((len(pattern_kpts2), 1))), axis=1).astype(np.float32)

        # 畸变矫正
        camera_matrix, dist_coefs, _rvecs, _tvecs, newcameramtx, dst, dst_full = calibrate(
            self.camera_img, camera_kpts, pattern_kpts, self.fisheye
        )
        # 单应性变换
        hom, dst_hom = find_hom(dst_full, self.pattern_img, pattern_kpts, _rvecs[0], _tvecs[0], newcameramtx)

        # show
        cv2.namedWindow("dst", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("dst_full", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("dst_hom", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("dst", dst)
        cv2.imshow("dst_full", dst_full)
        cv2.imshow("dst_hom", dst_hom)
        key = cv2.waitKey(0)

        if key == ord('r') and self.random_select != -1:
            self.random_select_kpts()
        else:
            # save
            cv2.imwrite(os.path.join(self.save_root, f"{name}_camera_select_kpts.jpg"), self.camera_show)
            cv2.imwrite(os.path.join(self.save_root, f"{name}_pattern_select_kpts.jpg"), self.pattern_show)
            np.savez(os.path.join(self.save_root, f"{name}_camera_select_kpts.npz"), kpts=camera_kpts)
            np.savez(os.path.join(self.save_root, f"{name}_pattern_select_kpts.npz"), kpts=pattern_kpts2)
            cv2.imwrite(os.path.join(self.save_root, f"{name}_result.jpg"), dst)
            cv2.imwrite(os.path.join(self.save_root, f"{name}_result_full.jpg"), dst_full)
            cv2.imwrite(os.path.join(self.save_root, f"{name}_hom.jpg"), dst_hom)
            np.savez(os.path.join(self.save_root, f"{name}_calibrate_parm.npz"),
                     camera_matrix=camera_matrix,
                     dist_coefs=dist_coefs,
                     _rvecs=_rvecs,
                     _tvecs=_tvecs)
            np.savez(os.path.join(self.save_root, f"{name}_hom.npz"), hom=hom)
            if self.random_select != -1:
                camera_kpts_raw = np.array(self.camera_kpts_select_raw, dtype=np.float32)
                pattern_kpts_raw = np.array(self.pattern_kpts_select_raw, dtype=np.float32)
                np.savez(os.path.join(self.save_root, f"{name}_camera_select_kpts_raw.npz"), kpts=camera_kpts_raw)
                np.savez(os.path.join(self.save_root, f"{name}_pattern_select_kpts_raw.npz"), kpts=pattern_kpts_raw)
            cv2.destroyAllWindows()

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
                    self.camera_kpts_select_raw = self.camera_kpts_select
                    self.pattern_kpts_select_raw = self.pattern_kpts_select
                    print("开始矫正")
                    self.calibrate()
                    break
                else:
                    print("相机与模板图取点数量不同或数量太少")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-img-path', type=str, default='data/nike/left_side_1.jpg', help='相机图像路径')
    parser.add_argument('--pattern-img-path', type=str, default='data/pattern/court.jpg', help='模板图路径')
    parser.add_argument('--camera-kpts-path', type=str, default=None, help='相机图像预选关键点')
    parser.add_argument('--pattern-kpts-path', type=str, default='data/pattern/court_kpts.npz', help='模板图像预选关键点')
    parser.add_argument('--camera-select-kpts-path', type=str, default='data/nike/left_side_1_camera_select_kpts.npz',
                        help='相机图像已选关键点')
    parser.add_argument('--pattern-select-kpts-path', type=str, default='data/nike/left_side_1_pattern_select_kpts.npz',
                        help='模板图像已选关键点')
    parser.add_argument('--fisheye', action='store_true', default=False, help='是否为鱼眼相机')
    parser.add_argument('--random-select', type=int, default=-1, help='随机取点数量')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    calibration = App(**vars(opt))
    calibration.run()
