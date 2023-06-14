import random
import numpy as np

def estimate_disp_sgm(left: np.ndarray, right: np.ndarray, maxdisp: int):
    """Compute disparity using SGM Opencv with (some) random settings"""
    p2_values = [32, 64, 96]
    block = [3, 5, 7]
    b = random.choice(block)
    p2 = random.choice(p2_values)
    sgm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=maxdisp,
        blockSize=b,
        uniquenessRatio=1,
        speckleWindowSize=0,
        speckleRange=0,
        disp12MaxDiff=200,
        P1=2 * b * b,
        P2=p2 * b * b,
    )
    left_p = np.copy(left)
    right_p = np.copy(right)

    # do some random augmentation
    left_p = color_aug(
        left_p,
        gamma_low=0.9,
        gamma_high=1.1,
        brightness_low=0.8,
        brightness_high=1.2,
        color_low=0.9,
        color_high=1.1,
        prob=0.2,
    )

    right_p = color_aug(
        right_p,
        gamma_low=0.9,
        gamma_high=1.1,
        brightness_low=0.8,
        brightness_high=1.2,
        color_low=0.9,
        color_high=1.1,
        prob=0.2,
    )
    left_p = left_p.astype(np.uint8)
    right_p = right_p.astype(np.uint8)

    left_p = np.pad(left_p, [(0, 0), (maxdisp, 0), (0, 0)])
    right_p = np.pad(right_p, [(0, 0), (maxdisp, 0), (0, 0)])

    disparity = sgm.compute(left_p, right_p) / 16.0
    disparity[disparity < 0] = 0
    disparity = disparity[:, maxdisp:]

    disparity = np.expand_dims(disparity, -1)
    return disparity


def estimate_disp_adcensus(left: np.ndarray, right: np.ndarray, maxdisp: int):
    """Compute disparity using AD-Census Opencv with random settings"""
    block_size = random.randrange(7, 21, step=2)
    uniqueness = random.randint(0, 15)
    ad_census = cv2.StereoBM_create(
        numDisparities=maxdisp,
        blockSize=block_size,
    )
    ad_census.setUniquenessRatio(uniqueness)
    left_p = np.copy(left)
    right_p = np.copy(right)

    # do some random augmentation
    left_p = color_aug(
        left_p,
        gamma_low=0.9,
        gamma_high=1.1,
        brightness_low=0.8,
        brightness_high=1.2,
        color_low=0.9,
        color_high=1.1,
        prob=0.2,
    )

    right_p = color_aug(
        right_p,
        gamma_low=0.9,
        gamma_high=1.1,
        brightness_low=0.8,
        brightness_high=1.2,
        color_low=0.9,
        color_high=1.1,
        prob=0.2,
    )
    left_p = left_p.astype(np.uint8)
    right_p = right_p.astype(np.uint8)

    # NOTE: AD-Census requires Grayscale images
    left_p = cv2.cvtColor(left_p, cv2.COLOR_BGR2GRAY)
    right_p = cv2.cvtColor(right_p, cv2.COLOR_BGR2GRAY)

    left_p = np.pad(left_p, [(0, 0), (maxdisp, 0)])
    right_p = np.pad(right_p, [(0, 0), (maxdisp, 0)])

    disparity = ad_census.compute(left_p, right_p) / 16.0
    disparity[disparity < 0] = 0
    disparity = disparity[:, maxdisp:]
    disparity = np.expand_dims(disparity, -1)
    return disparity