import os
import glob
import cv2
import numpy as np
import csv
from argparse import ArgumentParser


# def core_process(image, blur_radius=31, block_size=15, C=2, contours_keep=3):
def core_process(image):
    # image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, 255, 255)
    edges_inv = cv2.bitwise_not(edges)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    mask = cv2.bitwise_and(blurred_image, blurred_image, mask=edges_inv)
    mask = cv2.bitwise_or(mask, image, mask=edges)
    return mask


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

    mask = core_process(image)

    # 定义分析的区域
    x_start, x_end = 350, 500
    y_start, y_end = 780, 1030

    # 指定测量位置的Y坐标(中間點正負0.6mm)
    y_positions = [845, 910, 975]

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
    # def measure_widths_in_area(mask, y_positions, x_start, x_end, min_gap=15):
    #     measurements = []
    #     for y in y_positions:
    #         # 仅分析指定区域内的X坐标
    #         if y_start <= y <= y_end:
    #             x_indices = np.where(mask[y, x_start:x_end] == 255)[0] + x_start
    #             x_indices_list.append(x_indices)
    #             if verbose:
    #                 print(f"x_indices at y={y}: {x_indices}")
    #             if len(x_indices) >= 2:
    #                 # 外边缘宽度 D
    #                 D = x_indices[-1] - x_indices[0]
    #                 disc_index = [
    #                     i
    #                     for i in range(1, len(x_indices))
    #                     if x_indices[i] - x_indices[i - 1] > 1
    #                 ]
    #                 if len(disc_index) > 0:
    #                     disc = disc_index[0]
    #                     d = x_indices[disc] - x_indices[disc - 1]
    #                 else:
    #                     d = None
    #                 measurements.append((D, d))
    #             else:
    #                 measurements.append((None, None))
    #             disc_index_list.append(disc_index)
    #         else:
    #             measurements.append((None, None))
    #     return measurements

    def measure_widths_in_area(mask, y_positions, x_start, x_end, min_gap=15):
        measurements = []
        widths = []
        for y in y_positions:
            if y_start <= y <= y_end:
                x_indices = np.where(mask[y, x_start:x_end] > 100)[0] + x_start
                print("x_indices", x_indices)
                width = x_indices[-1] - x_indices[0]
                print("width", width)
                widths.append(width)
        return widths

    # 测量宽度
    # widths = measure_widths_in_area(mask, y_positions, x_start, x_end)
    # widths_d = [d for D, d in widths]
    # try:
    #     width_average = [(D + d) / 2 for D, d in widths]
    # except TypeError:
    #     print(f"Type Error, image {image_path} not processed")
    #     return
    #

    widths_d = measure_widths_in_area(mask, y_positions, x_start, x_end, min_gap=15)

    # 计算实际宽度
    # scale = 58.5 / 0.63  # = 92.85
    # scale10 = 91.3348  # average ratio (aoi/groundtruth) of 10 records
    scaleNew = 87.39
    scale = scaleNew

    # width_average_real = [w / scale for w in width_average]
    # width_average_real = [w / scale for w in width_average]
    width_d_real = [w / scale for w in widths_d]

    # limist to precision 4
    # width_average_real = [round(w, 4) for w in width_average_real]
    width_d_real = [round(w, 4) for w in width_d_real]

    if verbose:
        print("widths_d", widths_d)
        # print("width_average", width_average)
        # print("width_average_real", width_average_real)

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
                    "error1",
                    "error2",
                    "error3",
                ]
            )
        # 写入每个图像的测量数据
        # w1, w2, w3 = [float(e) for e in width_average_real]
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
            widths_d[0], widths_d[1], widths_d[2],
            error1, error2, error3,
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

    # for i, (D, d) in enumerate(widths):
    #     y = y_positions[i]
    #     if D is not None and d is not None:
    #         cv2.line(output_image, (x_start, y), (x_end, y), (0, 0, 255), 1)
    #         cv2.putText(
    #             output_image,
    #             f"D{i+1} = {D}px",
    #             (x_start + 5, y - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.6,
    #             (0, 0, 255),
    #             1,
    #         )
    #         cv2.putText(
    #             output_image,
    #             f"d{i+1} = {d}px",
    #             (x_start + 5, y + 20),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.6,
    #             (0, 0, 255),
    #             1,
    #         )

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


issues=[
    "images/bench_images/GL1H5F110EZ0000QTC+21111824421.jpg",
    "images/bench_images/GL1H5F1107E0000QTC+21112744A21.jpg",
]


for image_path in get_jpeg_file_paths(dir):
    # if "GL1H5E110480000QTC+21128873821.jpg" in image_path:
    # if "GL1H5F110F10000QTC+21157834721.jpg" in image_path:
    # if "GL1H5F110F90000QTC+21158544721.jpg" in image_path:
    # if "GL1H5E1104S0000QTC+21168144621.jpg" in image_path:
    # if "GL1H5E110480000QTC+21128873821.jpg" in image_path:
    print("image_path", image_path)
    if image_path not in issues:
        parse_image(outdir, groundtruth_data, image_path, show, show_early, verbose)
