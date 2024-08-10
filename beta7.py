import os
import cv2
import numpy as np
import csv
from argparse import ArgumentParser


def parse_image(outdir, groundtruth_data, image_path=None, show=False, verbose=False):
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

    # 使用 Gaussian 模糊来平滑图像，减少噪声
    blur_radius = 13
    image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

    # 使用自适应阈值来进行二值化
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 反转颜色，使得白色部分为我们感兴趣的区域
    binary = cv2.bitwise_not(binary)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 选择最大的轮廓进行简化和平滑
    contours_keep = 4
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:contours_keep]

    # 选择最内层的轮廓来绘制
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # 定义分析的区域
    x_start, x_end = 350, 500
    y_start, y_end = 780, 1030

    # 指定测量位置的Y坐标(中間點正負0.6mm)
    y_positions = [845, 910, 975]

    x_indices_list = []
    disc_index_list = []

    # 仅分析指定区域内的这些Y坐标
    def measure_widths_in_area(mask, y_positions, x_start, x_end, min_gap=15):
        measurements = []
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
                        if x_indices[i] - x_indices[i - 1] > 1
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
    width_average = [(D + d) / 2 for D, d in widths]

    # 计算实际宽度
    scale = 58.5 / 0.63  # = 92.85
    scale10 = 91.3348 # average ratio (aoi/groundtruth) of 10 records

    # width_average_real = [w / scale for w in width_average]
    width_average_real = [w / scale10 for w in width_average]

    # limist to precision 4
    width_average_real = [round(w, 4) for w in width_average_real]

    if verbose:
        print("widths", widths)
        print("width_average", width_average)
        print("width_average_real", width_average_real)

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
                    "W1", "W2", "W3",
                    "error1", "error2", "error3",
                    "D1", "d1",
                    "D2", "d2",
                    "D3", "d3",
                ]
            )
        # 写入每个图像的测量数据
        gt1, gt2, gt3 = [float(e) for e in gt_values]
        w1, w2, w3 = [float(e) for e in width_average_real]
        # import ipdb; ipdb.set_trace()
        row = [
            imagename,
            gt1, gt2, gt3,
            w1, w2, w3,
            width_average[0], width_average[1], width_average[2],
            round((w1 - gt1) / gt1 * 100, 1),
            round((w2 - gt2) / gt2 * 100, 1),
            round((w3 - gt3) / gt3 * 100, 1),
            widths[0][0], widths[0][1],
            widths[1][0], widths[1][1],
            widths[2][0], widths[2][1],
        ]
        writer.writerow(row)

    # 在图像上绘制测量结果
    output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

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
            if verbose:
                print(f"Dx1={Dx1}, Dx2={Dx2}, x1={x1}, x2={x2}")
            cv2.line(output_image, (x1, y - height), (x1, y + height), (255, 0, 0), 1)
            cv2.line(output_image, (x2, y - height), (x2, y + height), (255, 0, 0), 1)

    for i, (D, d) in enumerate(widths):
        y = y_positions[i]
        if D is not None and d is not None:
            cv2.line(output_image, (x_start, y), (x_end, y), (0, 0, 255), 1)
            cv2.putText(
                output_image,
                f"D{i+1} = {D}px",
                (x_start + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                output_image,
                f"d{i+1} = {d}px",
                (x_start + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )

    # 在图像上绘制矩形区域
    cv2.rectangle(output_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

    # 裁剪图像仅保留感兴趣区域
    cropped_image = output_image[y_start:y_end, x_start:x_end]

    if show:
        cv2.imshow("Measurement Results", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 保存结果图像
    cv2.imwrite(f"{outdir}/{imagename}.png", cropped_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--show", action="store_true", help="Display the result images")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed processing information"
    )
    args = parser.parse_args()

    outdir = args.outdir
    show = args.show
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

    imagepaths = [
        "images/bench_images/GL135141V1L0XFCA1+P13434443114.jpg",
        "images/bench_images/GL135141V3E0XFCA1+P13152515314.jpg",
        "images/bench_images/GL135141V6X0XFCA9+P12276145B14.jpg",
        "images/bench_images/GL135141VA80XFCAL+P13126776514.jpg",
        "images/bench_images/GL135141W1X0XFCAP+P13417553114.jpg",
        "images/bench_images/GL135141W4U0XFCAH+P12163853114.jpg",
        "images/bench_images/GL135141X2S0XFCAQ+P13437334614.jpg",
        "images/bench_images/GL135141X3S0XFCAM+P13148724914.jpg",
        "images/bench_images/GL135141X3T0XFCAL+P13224253614.jpg",
        "images/bench_images/GL135141X4N0XFCAN+P13284764B14.jpg",
    ]

for image_path in imagepaths:
    parse_image(outdir, groundtruth_data, image_path, show, verbose)
