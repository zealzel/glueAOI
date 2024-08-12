import os
import glob
import cv2
import numpy as np
import csv
from argparse import ArgumentParser


def core_process(image, blur_radius=31, block_size=15, C=2, contours_keep=3):
    blur_radius, block_size, C, contours_keep = 31, 21, 2, 4  # <-- new default
    # blur_radius, block_size, C, contours_keep = 31, 15, 2, 3 # <-- new default
    # blur_radius, block_size, C, contours_keep = 13, 11, 2, 4  # <--- beta6

    # # 使用 Gaussian 模糊来平滑图像，减少噪声
    # image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

    # # 使用自适应阈值来进行二值化
    # binary = cv2.adaptiveThreshold(
    #     image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    # )
    blured_image = cv2.medianBlur(image, 5)
    _, binary = cv2.threshold(blured_image, 200, 255,cv2.THRESH_BINARY)

    # 反转颜色，使得白色部分为我们感兴趣的区域
    binary = cv2.bitwise_not(binary)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 选择最大的轮廓进行简化和平滑
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:contours_keep]

    # 选择最内层的轮廓来绘制
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask, contours


def parse_image(
    outdir,
    groundtruth_data,
    image_path=None,
    show=False,
    show_early=False,
    verbose=False,
):
    if not image_path:
        print("image_path is None")
        return

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    imagename = os.path.splitext(os.path.basename(image_path))[0]
    if verbose:
        print(f"\nProcessing image: {imagename}")

    # 检查是否已经存在记录
    csv_path = os.path.join(outdir, "output.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, mode="r") as file:
            reader = csv.reader(file)
            existing_imagenames = [row[0] for row in reader if row]
            if imagename in existing_imagenames:
                if verbose:
                    print(f"{imagename} 已存在，跳过.")
                return

    mask, contours = core_process(image)

    # 定义分析的区域
    x_start, x_end = 350, 500
    y_start, y_end = 780, 1030

        # 计算实际宽度
    scaleNew = 88.161 # 30 average
    scale = scaleNew
    # 指定测量位置的Y坐标(中間點正負0.6mm)
    offset = 0.6*scale
    center = 910
    y_positions = [round(center-offset), center, round(center+offset)]

    x_indices_list = []
    disc_index_list = []

    if show_early:
        cv2.namedWindow("show_early", cv2.WINDOW_NORMAL)
        cropped_image = mask[y_start:y_end, x_start:x_end]
        cv2.imshow("show_early", cropped_image)
        # cv2.imshow("Measurement Results", mask)
        cv2.resizeWindow("show_early", 500, 800)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 仅分析指定区域内的这些Y坐标
    def measure_widths_in_area(mask, y_positions, x_start, x_end, min_gap=15):
        measurements = []
        disc_threshold = 5
        for y in y_positions:
            # 仅分析指定区域内的X坐标
            if y_start <= y <= y_end:
                x_indices = np.where(mask[y, x_start:x_end] == 255)[0] + x_start
                x_indices_list.append(x_indices)
                if verbose:
                    print(f"x_indices at y={y}: {x_indices}")
                if len(x_indices) >= 2:
                    # 外边缘宽度 D
                    D = x_indices[-1] - x_indices[0]
                    disc_index = [
                        i
                        for i in range(1, len(x_indices))
                        if x_indices[i] - x_indices[i - 1] > disc_threshold
                    ]
                    if len(disc_index) > 0:
                        disc = disc_index[0]
                        d = x_indices[disc] - x_indices[disc - 1]
                    else:
                        d = None
                    measurements.append((D, d))
                else:
                    measurements.append((None, None))
                disc_index_list.append(disc_index)
            else:
                measurements.append((None, None))
        return measurements

    # 测量宽度
    widths = measure_widths_in_area(mask, y_positions, x_start, x_end)
    widths_d = [d for D, d in widths]
    try:
        width_average = widths_d
    except TypeError:
        print(f"Type Error, image {image_path} not processed")
        return


    # width_average_real = [w / scale for w in width_average]
    width_average_real = [w / scale for w in width_average]
    width_d_real = [w / scale for w in widths_d]

    # limist to precision 4
    width_average_real = [round(w, 4) for w in width_average_real]
    width_d_real = [round(w, 3) for w in width_d_real]

    if verbose:
        print("widths", widths)

    # 获取 ground truth 数据
    gt_values = groundtruth_data.get(imagename, ["N/A", "N/A", "N/A"])

    # 保存结果到 CSV 文件
    csv_path = os.path.join(outdir, "output.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # 写入CSV文件的表头
            writer.writerow(
                [
                    "imagename",
                    "gt1", "gt2", "gt3",
                    "w1", "w2", "w3",
                    "error1", "error2", "error3",
                    "d1", "d2", "d3",
                ]
            )
        # 写入每个图像的测量数据
        w1, w2, w3 = [float(e) for e in width_d_real]

        try:
            gt1, gt2, gt3 = [float(e) for e in gt_values]
            error1, error2, error3 = [
                round((w1 - gt1) / gt1 * 100, 1),
                round((w2 - gt2) / gt2 * 100, 1),
                round((w3 - gt3) / gt3 * 100, 1),
            ]
        except ValueError:
            gt1 = gt2 = gt3 = error1 = error2 = error3 = "N/A"

        # import ipdb; ipdb.set_trace()
        row = [
            imagename,
            gt1, gt2, gt3,
            w1, w2, w3,
            error1, error2, error3,
            widths[0][1], widths[1][1], widths[2][1],
        ]
        writer.writerow(row)

    # 在图像上绘制测量结果
    output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image0 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i, x_indices in enumerate(x_indices_list):
        height = 5
        disc_index = disc_index_list[i]
        y = y_positions[i]
        Dx1, Dx2 = x_indices[0], x_indices[-1]
        cv2.line(output_image, (Dx1, y - height), (Dx1, y + height), (255, 0, 0), 1)
        cv2.line(output_image, (Dx2, y - height), (Dx2, y + height), (255, 0, 0), 1)

        if disc_index:
            idx = disc_index[0]
            x1, x2 = x_indices[idx - 1], x_indices[idx]
            # if verbose:
            #     print(f"Dx1={Dx1}, Dx2={Dx2}, x1={x1}, x2={x2}")
            cv2.line(output_image, (x1, y - height), (x1, y + height), (255, 0, 0), 1)
            cv2.line(output_image, (x2, y - height), (x2, y + height), (255, 0, 0), 1)

    for i, (D, d) in enumerate(widths):
        y = y_positions[i]
        if D is not None and d is not None:
            cv2.line(output_image, (x_start, y), (x_end, y), (0, 0, 255), 1)
            cv2.putText(output_image, f"w{i+1}={width_d_real[i]}mm", (x_start + 55, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(output_image, f"d{i+1}={d}px", (x_start + 55, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 在图像上绘制矩形区域
    cv2.rectangle(output_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)
    cropped_image = output_image[y_start:y_end, x_start:x_end]
    cropped_image0 = image0[y_start:y_end, x_start:x_end]

    # cv2.namedWindow("show", cv2.WINDOW_NORMAL)
    # cv2.imshow("show", cropped_image)
    # cv2.resizeWindow("show", 450, 650)
    # cv2.namedWindow("show0", cv2.WINDOW_NORMAL)
    # cv2.imshow("show0", cropped_image0)
    # cv2.resizeWindow("show0", 450, 650)
    combined_image = cv2.hconcat([cropped_image, cropped_image0])
    cv2.namedWindow("combined_show", cv2.WINDOW_NORMAL)
    cv2.putText(combined_image, f"{imagename}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if show:
        cv2.imshow("combined_show", combined_image)
        cv2.resizeWindow("combined_show", 1000, 800)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 保存结果图像
    # cv2.imwrite(f"{outdir}/{imagename}.png", cropped_image)
    cv2.imwrite(f"{outdir}/{imagename}.png", combined_image)


# 取得指定資料夾下所有圖片的路徑
def get_jpeg_file_paths(folder_path):
    for file_path in glob.iglob(
        os.path.join(folder_path, "**", "GL*.jpg"), recursive=True
    ):
        yield file_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="images/bench_images")
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--show", action="store_true", help="Display the result images")
    parser.add_argument(
        "--show_early", action="store_true", help="Show the core process image first"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed processing information"
    )
    args = parser.parse_args()

    dir = args.dir
    outdir = args.outdir
    show = args.show
    show_early = args.show_early
    verbose = args.verbose

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 读取 groundtruth.csv
    groundtruth_path = "groundtruth.csv"
    groundtruth_data = {}
    if os.path.isfile(groundtruth_path):
        with open(groundtruth_path, mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                imagename, gt1, gt2, gt3 = row[0], row[1], row[2], row[3]
                groundtruth_data[imagename] = [gt1, gt2, gt3]
    else:
        if verbose:
            print(f"No {groundtruth_path} detected")


bigerr = [
    # "images/bench_images/GL1H5E1106J0000QTC+21146523B21.jpg",
    # "images/bench_images/GL1H5E1103Y0000QTC+21132144A21.jpg",
    # "images/bench_images/GL1H5E110480000QTC+21128873821.jpg",
    # "images/bench_images/GL1H5E1102T0000QTC+21132823B21.jpg",
    # "images/bench_images/GL1H5F110B20000QTC+21136625821.jpg",
    # "images/bench_images/GL1H5F110CD0000QTC+21172545B21.jpg",
    # "images/bench_images/GL1H5F110AQ0000QTC+21141575821.jpg",
    # "images/bench_images/GL1H5F110BB0000QTC+21165653721.jpg",
    # "images/bench_images/GL1H5E1106M0000QTC+21185136221.jpg",
    # "images/bench_images/GL1H5F110FP0000QTC+21265365521.jpg",
    #
    "images/bench_images/GL1H5E1103Y0000QTC+21132144A21.jpg",
    "images/bench_images/GL1H5E110480000QTC+21128873821.jpg",
    "images/bench_images/GL1H5F110B20000QTC+21136625821.jpg",
    "images/bench_images/GL1H5F110CD0000QTC+21172545B21.jpg",
    "images/bench_images/GL1H5F110AQ0000QTC+21141575821.jpg",
    "images/bench_images/GL1H5F110BB0000QTC+21165653721.jpg",
    "images/bench_images/GL1H5E1106M0000QTC+21185136221.jpg",
    "images/bench_images/GL1H5F110FP0000QTC+21265365521.jpg",
]

# for image_path in get_jpeg_file_paths("images/bench_images/"):
for image_path in get_jpeg_file_paths(dir):
    # if image_path in bigerr:
    print("image_path", image_path)
    parse_image(outdir, groundtruth_data, image_path, show, show_early, verbose)

    # ==== snippts ====
    # show aoi image one by one
    # python alpha1.py --verbose --show
    #
    # only show preliminary edge-find image one by one, no measure
    # python alpha1.py --verbose --show_early
