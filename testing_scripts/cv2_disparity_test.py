import cv2
import os
import numpy as np

from matplotlib import pyplot as plt

# MAX_DISP = {"chair": 256, "cones": 64, "teddy": 64}
MAX_DISP = {"chair": 256, "cones":64, "teddy": 64}


def show(pred_disp, disp_left, disp_right, algorithm):
    plt.clf()
    plt.title(f"Left Disparity with {algorithm}")
    plt.imshow(pred_disp, 'gray')
    plt.show()

    plt.clf()
    plt.title(f"Left Disparity GT")
    plt.imshow(disp_left, 'gray')
    plt.show()

    plt.clf()
    plt.title(f"Right Disparity GT")
    plt.imshow(disp_right, 'gray')
    plt.show()


def disparity(dataset, algorithm, show_img=True, output_dim=(320, 320)):
    max_disp = MAX_DISP[dataset]

    disp_left_path = os.path.join("samples", dataset, "left_disp.png")
    disp_right_path = os.path.join("samples", dataset, "right_disp.png")
    rgb_left_path = os.path.join("samples", dataset, "left_rgb.png")
    rgb_right_path = os.path.join("samples", dataset, "right_rgb.png")

    disp_left = cv2.imread(disp_left_path, cv2.IMREAD_GRAYSCALE)
    disp_right = cv2.imread(disp_right_path, cv2.IMREAD_GRAYSCALE)

    disp_left = cv2.resize(src=disp_left, dsize=output_dim)
    disp_right = cv2.resize(src=disp_right, dsize=output_dim)

    rgb_left = cv2.imread(rgb_left_path, cv2.IMREAD_GRAYSCALE)
    rgb_right = cv2.imread(rgb_right_path, cv2.IMREAD_GRAYSCALE)

    # rgb_left = cv2.resize(src=rgb_left, dsize=output_dim)
    # rgb_right = cv2.resize(src=rgb_right, dsize=output_dim)

    rgb_left = np.pad(array=rgb_left, pad_width=[(0, 0), (max_disp, 0)])
    rgb_right = np.pad(array=rgb_right, pad_width=[(0, 0), (max_disp, 0)])

    if algorithm == "BM":
        stereo = cv2.StereoBM_create(numDisparities=max_disp, blockSize=15)
        # divide by 16 from: https://stackoverflow.com/questions/27856965/stereo-disparity-map-generation
        pred_disp = stereo.compute(rgb_left, rgb_right) / 16.0
        pred_disp[pred_disp < 0] = 0
        pred_disp = pred_disp[:, max_disp:]
        rgb_left = rgb_left[:, max_disp:]

        # pred_disp = pred_disp[:output_dim[0], :output_dim[1]]
        # rgb_left = rgb_left[:output_dim[0], :output_dim[1]]

        if show_img:
            show(pred_disp=pred_disp, disp_left=disp_left, disp_right=disp_right, algorithm=algorithm)

    elif algorithm == "SGM":

        # p2_values = [32, 64, 96]
        # block = [3, 5, 7]
        # b = random.choice(block)
        # p2 = random.choice(p2_values)
        b = 7
        p2 = 96

        sgm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max_disp,
            blockSize=7,
            uniquenessRatio=1,
            speckleWindowSize=0,
            speckleRange=0,
            disp12MaxDiff=max_disp,
            P1=2 * b * b,
            P2=p2 * b * b
        )

        # sgm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max_disp)
        pred_disp = sgm.compute(rgb_left, rgb_right) / 16.0
        pred_disp[pred_disp < 0] = 0
        pred_disp = pred_disp[:, max_disp:]
        rgb_left = rgb_left[:, max_disp:]

        # pred_disp = pred_disp[:output_dim[0], :output_dim[1]]
        # rgb_left = rgb_left[:output_dim[0], :output_dim[1]]

        if show_img:
            show(pred_disp=pred_disp, disp_left=disp_left, disp_right=disp_right, algorithm=algorithm)

    elif algorithm == "AC":
        # block_size = random.randrange(7, 21, step=2)
        # uniqueness = random.randint(0, 15)
        uniqueness = 7
        ad_census = cv2.StereoBM_create(numDisparities=max_disp, blockSize=15)
        ad_census.setUniquenessRatio(uniqueness)
        pred_disp = ad_census.compute(rgb_left, rgb_right) / 16.0
        pred_disp[pred_disp < 0] = 0
        pred_disp = pred_disp[:, max_disp:]
        rgb_left = rgb_left[:, max_disp:]

        # pred_disp = pred_disp[:output_dim[0], :output_dim[1]]
        # rgb_left = rgb_left[:output_dim[0], :output_dim[1]]

        if show_img:
            show(pred_disp=pred_disp, disp_left=disp_left, disp_right=disp_right, algorithm=algorithm)

    else:
        raise ValueError("Algorithm is not valid")

    print(f"Original shape: {disp_left.shape}, Predicted shape: {pred_disp.shape}")

    pred_disp = cv2.resize(pred_disp, dsize=output_dim)


    rgb_left = cv2.imread(rgb_left_path)

    rgb_left = cv2.resize(rgb_left, dsize=output_dim)

    return pred_disp.astype(np.uint8), rgb_left.astype(np.uint8)


if __name__ == '__main__':
    # for dataset in ["cones", "chair", "teddy"]:
    # # for dataset in ["cones"]:
    #     pred_disp, rgb_left = disparity(dataset=dataset, algorithm="BM", show_img=False)
    #     cv2.imwrite(f"samples/outputs/{dataset}_BM_disp.png", pred_disp)
    #     cv2.imwrite(f"samples/outputs/{datasRet}_BM_rgb.png", rgb_left)
    #     plt.imshow(pred_disp, "gray")
    #     # plt.show()
    #     # print(np.max(prRed_disp))
    #     # plt.savefig(f"samples/outputs/PLT_{dataset}_BM_disp.png")
    #     pred_disp, rgb_left = disparity(dataset=dataset, algorithm="SGM", show_img=False)
    #     cv2.imwrite(f"samples/outputs/{dataset}_SGM_disp.png", pred_disp)
    #     cv2.imwrite(f"samples/outputs/{dataset}_SGM_rgb.png", rgb_left)
    #     pred_disp, rgb_left = disparity(dataset=dataset, algorithm="AC", show_img=False)
    #     cv2.imwrite(f"samples/outputs/{dataset}_AC_disp.png", pred_disp)
    #     cv2.imwrite(f"samples/outputs/{dataset}_AC_rgb.png", rgb_left)


    disp = cv2.imread("samples/teddy/left_disp.png", cv2.IMREAD_GRAYSCALE)

    disp = cv2.resize(disp, dsize=(320, 320))

    edges = cv2.Laplacian(disp, -1, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite(f"samples/teddy/left_disp_edge.png", edges)