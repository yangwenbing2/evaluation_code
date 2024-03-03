import os
import cv2
import argparse

from evaluations import calculate_sad_mse_mad_whole_img, compute_mse_loss, compute_sad_loss, compute_mad_loss
from evaluations import compute_gradient_loss, compute_connectivity_error


def read_image(name, args):
    # read the alpha
    alpha = cv2.imread(os.path.join(args.alpha_dir, name)) / 255.
    alpha = alpha[:, :, 0] if alpha.ndim > 2 else alpha

    # read the trimap
    trimap = cv2.imread(os.path.join(args.trimap_dir, name))
    trimap = trimap[:, :, 0] if trimap.ndim > 2 else trimap
    trimap[(0 < trimap) & (trimap < 255)] = 1
    trimap[trimap == 255] = 2

    # read the pred
    pred = cv2.imread(os.path.join(args.pred_dir, name.replace(".jpg", ".png"))) / 255.
    pred = pred[:, :, 0] if pred.ndim > 2 else pred

    return alpha, trimap, pred


def main(args):
    sad_loss, mse_loss, mad_loss, grad_loss, conn_loss = 0, 0, 0, 0, 0
    sad_trimap_loss, mse_trimap_loss, mad_trimap_loss = 0, 0, 0
    names = os.listdir(args.alpha_dir)

    count = len(names)
    for idx, name in enumerate(names):
        print("Processing:[{}/{}]   name:{}".format(idx + 1, count, name))

        if not name[-4:] in [".png", ".jpg"]:
            count -= 1
            continue

        alpha, trimap, pred = read_image(name, args)

        sad_diff = compute_sad_loss(pred, alpha)
        mse_diff = compute_mse_loss(pred, alpha)
        mad_diff = compute_mad_loss(pred, alpha)
        grad_diff = compute_gradient_loss(pred, alpha)
        conn_diff = compute_connectivity_error(pred, alpha)

        sad_trimap_diff = compute_sad_loss(pred, alpha, trimap)
        mse_trimap_diff = compute_mse_loss(pred, alpha, trimap)
        mad_trimap_diff = compute_mad_loss(pred, alpha, trimap)

        sad_loss += sad_diff
        mse_loss += mse_diff
        mad_loss += mad_diff
        grad_loss += grad_diff
        conn_loss += conn_diff

        sad_trimap_loss += sad_trimap_diff
        mse_trimap_loss += mse_trimap_diff
        mad_trimap_loss += mad_trimap_diff

    print("\n-------------------- Result ----------------------")
    print("sad_loss:{:.6f}".format(sad_loss / count))
    print("mse_loss:{:.6f}".format(mse_loss / count))
    print("mad_loss:{:.6f}".format(mad_loss / count))
    print("grad_loss:{:.6f}".format(grad_loss / count))
    print("conn_loss:{:.6f}".format(conn_loss / count))

    print("sad_trimap_loss:{:.6f}".format(sad_trimap_loss / count))
    print("mse_trimap_loss:{:.6f}".format(mse_trimap_loss / count))
    print("mad_trimap_loss:{:.6f}".format(mad_trimap_loss / count))

    name = args.pred_dir.split("/")[-2] + ".txt"
    result_path = os.path.join("./results", "debug.txt")
    with open(result_path, "w") as fb:
        fb.write("sad_loss:{:.6f}\nmse_loss:{:.6f}\nmad_loss:{:.6f}\n".format(sad_loss / count,
                                                                            mse_loss / count, mad_loss / count))
        fb.write("grad_loss:{:.6f}\nconn_loss:{:.6f}\n\n".format(grad_loss / count, conn_loss / count))
        fb.write("sad_trimap_loss:{:.6f}\nmse_trimap_loss:{:.6f}\nmad_trimap_loss:{:.6f}".format(
            sad_trimap_loss / count, mse_trimap_loss / count, mad_trimap_loss / count))
        print("Save successfully!!!")


def get_args():
    parser = argparse.ArgumentParser()

    # ========================================== HHM-17K ==========================================
    parser.add_argument('--alpha_dir', type=str, default='/media/ilab/Innocent/ilab/experiments/datasets/HHM-17k/test'
                                                         '/alpha')
    parser.add_argument('--trimap_dir', type=str, default='/media/ilab/Innocent/ilab/experiments/datasets/HHM-17k'
                                                          '/test/trimap')
    parser.add_argument('--pred_dir', type=str, default='/media/ilab/Innocent/ilab/experiments/MyNet/baseline_Unet'
                                                        '/prediction/b1_3_HHM-17K/resize')

    # Parse configuration
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
