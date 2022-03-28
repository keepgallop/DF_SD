'''
@Description  : Calculate visual similarity in terms of SSIM and PSNR
@Author       : Chi Liu
@Date         : 2022-03-02 16:20:59
@LastEditTime : 2022-03-03 17:48:40
'''
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def psnr_single(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "im1 and im2 should have the same shape"
    assert im1.dtype == im2.dtype, "im1 and im2 should have the same dtype"
    R = 255 if im1.dtype == int else 1
    psnr_score = cv2.PSNR(im1, im2, R=R)
    return psnr_score


def ssim_single(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "im1 and im2 should have the same shape"
    assert im1.dtype == im2.dtype, "im1 and im2 should have the same dtype"
    data_range = 255 if im1.dtype == int else 1
    # assert im2.dtype == float
    if im1.ndim == 3 and im2.ndim == 3:  # RGB images
        ssim_score = ssim(
            im1, im2, multichannel=True, data_range=data_range, gaussian_weights=True
        )
    # gray images, recommended for higher ssim scores.
    elif im1.ndim == 2 and im2.ndim == 2:
        ssim_score = ssim(
            im1, im2, multichannel=False, data_range=data_range, gaussian_weights=True
        )
    return ssim_score


def psnr_batch(imb1: np.ndarray, imb2: np.ndarray):
    """PSNR scores for an image batch

    Args:
        imb1 (np.ndarray):  N*W*H*D
        imb2 (np.ndarray):  N*W*H*D

    Returns:
        mean psnr score, psnr score list
    """
    psnr_scores = np.array([psnr_single(i[0], i[1]) for i in zip(imb1, imb2)])
    return np.mean(psnr_scores), psnr_scores


def ssim_batch(imb1: np.ndarray, imb2: np.ndarray):
    """SSIM scores for an image batch

    Args:
        imb1 (np.ndarray):  N*W*H*D
        imb2 (np.ndarray):  N*W*H*D

    Returns:
        mean ssim score, ssim score list
    """
    ssim_scores = np.array([ssim_single(i[0], i[1]) for i in zip(imb1, imb2)])
    return np.mean(ssim_scores), ssim_scores


def print_similarity_results(imb1: np.ndarray, imb2: np.ndarray):
    mean_psnr, _ = psnr_batch(imb1, imb2)
    mean_ssim, _ = ssim_batch(imb1, imb2)
    print(
        f"Similarity results => mean PSNR: {mean_psnr:.4f};  mean SSIM: {mean_ssim:.4f}"
    )


if __name__ == "__main__":
    # RGB test
    rgb_test_im1 = np.random.randint(0, 255, (100, 128, 128, 3))
    rgb_test_im2 = np.random.randint(0, 255, (100, 128, 128, 3))

    print(rgb_test_im1.dtype)
    rgb_test_im3 = rgb_test_im1 / 255.0
    rgb_test_im4 = rgb_test_im2 / 255.0
    print(rgb_test_im3.dtype)
    print_similarity_results(rgb_test_im1, rgb_test_im2)
    print_similarity_results(rgb_test_im3, rgb_test_im4)

    # gray test
    gray_test_im1 = np.random.randint(0, 255, (100, 128, 128))
    gray_test_im2 = np.random.randint(0, 255, (100, 128, 128))

    gray_test_im3 = gray_test_im1 / 255.0
    gray_test_im4 = gray_test_im2 / 255.0

    print_similarity_results(gray_test_im1, gray_test_im2)
    print_similarity_results(gray_test_im3, gray_test_im4)

    t1 = np.random.randint(0, 255, (128, 128, 3)) / 1.0
    t2 = np.random.randint(0, 255, (128, 128, 3)) / 1.0
    print(psnr_single(t1, t2))
