import cv2
import numpy as np

# 读取图像
image_path = "images/bench_images/GL135141V1L0XFCA1+P13434443114.jpg"
# image_path = "images/bench_images/GL135141V3E0XFCA1+P13152515314.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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
x_start, x_end = 350, 450
y_start, y_end = 800, 1000

# 指定测量位置的Y坐标
# y_positions = [830, 900, 980]
y_positions = [840, 900, 980]


p1, radius = (x_start, y_start), 10
p2, radius = (x_end, y_end), 10


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
            print("x_indices", x_indices)
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

# 在图像上绘制测量结果
output_image = cv2.cvtColor(
    mask, cv2.COLOR_GRAY2BGR
)  # 将灰度图像转换为BGR以便绘制彩色线条

for i, x_indices in enumerate(x_indices_list):
    height = 5
    disc_index = disc_index_list[i]
    # 在每个不连续点位置画纵线
    y = y_positions[i]
    Dx1, Dx2 = x_indices[-1], x_indices[0]
    cv2.line(output_image, (Dx1, y - height), (Dx1, y + height), (255, 0, 0), 1)
    cv2.line(output_image, (Dx2, y - height), (Dx2, y + height), (255, 0, 0), 1)
    print("Dx1, Dx2", Dx1, Dx2)
    for idx in disc_index:
        x1, x2 = x_indices[idx], x_indices[idx - 1]
        print("x1, x2", x1, x2)
        # print(y - height, y + height)
        cv2.line(output_image, (x1, y - height), (x1, y + height), (255, 0, 0), 1)
        cv2.line(output_image, (x2, y - height), (x2, y + height), (255, 0, 0), 1)


for i, (D, d) in enumerate(widths):
    y = y_positions[i]
    if D is not None and d is not None:
        cv2.line(
            output_image, (x_start, y), (x_end, y), (0, 0, 255), 1
        )  # 画出 y 坐标的水平线
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


scale = 58.5 / 0.63  # = 92.85
width_average_real = [w / scale for w in width_average]

# 显示结果
print("widths", widths)
print("width_average", width_average)
print("width_average_real", width_average_real)


cv2.rectangle(output_image, p1, p2, (0, 255, 0), 1)

cv2.imshow("Measurement Results", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite("xxx.png", output_image)
