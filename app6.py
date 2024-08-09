import cv2
import numpy as np

# 读取图像
image_path = "/mnt/data/image.png"
image_path = "images/GL135141V1L0XFCA1+P13434443114-Normal.jpg"
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

# 指定测量位置的Y坐标
y_positions = [850, 900, 950]


p1,radius = (300,600), 20
p2,radius = (600,1200), 20
cv2.circle(mask, p1, radius, 255, -1)
cv2.circle(mask, p2, radius, 255, -1)
p1,radius = (300,600), 20
p2,radius = (600,1200), 20
cv2.circle(mask, p1, radius, 255, -1)
cv2.circle(mask, p2, radius, 255, -1)

# 计算左侧的内外边缘的宽度
def measure_widths(mask, y_positions, min_gap=15):
    measurements = []
    all_x_indices = []
    threshold = 200
    for y in y_positions:
        x_indices = np.where(mask[y, :] >= threshold)[0]
        # import pdb; pdb.set_trace()
        all_x_indices.append(x_indices)
        if len(x_indices) >= 2:
            # 外边缘宽度 D
            D = x_indices[-1] - x_indices[0]

            # 内边缘宽度 d - 忽略过于接近的内外边缘点
            inner_indices = x_indices[1:-1]
            if len(inner_indices) > 1:
                valid_inner_indices = [
                    x
                    for x in inner_indices
                    if x - x_indices[0] > min_gap and x_indices[-1] - x > min_gap
                ]
                if len(valid_inner_indices) >= 2:
                    d = valid_inner_indices[-1] - valid_inner_indices[0]
                else:
                    d = None
            else:
                d = None

            measurements.append((D, d))
        else:
            measurements.append((None, None))
    return measurements, all_x_indices


# 测量宽度
widths, all_x_indices = measure_widths(mask, y_positions)

# 在图像上绘制测量结果
output_image = cv2.cvtColor(
    mask, cv2.COLOR_GRAY2BGR
)  # 将灰度图像转换为BGR以便绘制彩色线条

width_average = [(D + d) / 2 for D, d in widths]

for i, (D, d) in enumerate(widths):
    y = y_positions[i]
    x_indices = all_x_indices[i]
    if D is not None and d is not None:
        cv2.line(
            output_image, (0, y), (mask.shape[1], y), (0, 0, 255), 1
        )  # 画出 y 坐标的水平线
        cv2.putText(
            output_image,
            f"D{i+1} = {D}px",
            (x_indices[0] + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            output_image,
            f"d{i+1} = {d}px",
            (x_indices[1] + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )
    elif D is not None:
        cv2.putText(
            output_image,
            f"D{i+1} = {D}px",
            (x_indices[0] + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            output_image,
            f"d{i+1} = N/A",
            (x_indices[1] + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )

# 显示结果
print("widths", widths)
print("width_average", width_average)

cv2.imshow("Measurement Results", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
# cv2.imwrite("/mnt/data/output_measurement_improved.png", output_image)

