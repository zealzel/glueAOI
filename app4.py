import cv2
import numpy as np

# 讀取圖片
image_path = "images/GL135141V1L0XFCA1+P13434443114-Normal.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 使用 Gaussian 模糊來平滑圖像，減少雜訊
blur_radius = 13
image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

# 使用自適應閾值來進行二值化
binary = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# 反轉顏色，使得白色部分為我們感興趣的區域
binary = cv2.bitwise_not(binary)

# 尋找輪廓
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 選擇最大的輪廓進行簡化和平滑
contours_keep = 4
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:contours_keep]

# 選擇最內層的輪廓來繪製
mask = np.zeros_like(image)
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# 指定量測位置的Y座標
y_positions = [850, 900, 950]


# 計算左側的內外邊緣的寬度
def measure_widths(mask, y_positions):
    measurements = []
    all_x_indices = []
    for y in y_positions:
        x_indices = np.where(mask[y, :] == 255)[0]
        all_x_indices.append(x_indices)
        if len(x_indices) >= 2:
            D = x_indices[-1] - x_indices[0]  # 外邊緣寬度
            d = x_indices[-2] - x_indices[1]  # 內邊緣寬度
            measurements.append((D, d))
        else:
            measurements.append((None, None))
    return measurements, all_x_indices


# 量測寬度
widths, all_x_indices = measure_widths(mask, y_positions)

# 在圖片上繪製測量結果
output_image = cv2.cvtColor(
    mask, cv2.COLOR_GRAY2BGR
)  # 將灰度圖像轉換為BGR以便繪製彩色線條

for i, (D, d) in enumerate(widths):
    y = y_positions[i]
    x_indices = all_x_indices[i]
    if D is not None and d is not None:
        cv2.line(
            output_image, (0, y), (mask.shape[1], y), (0, 0, 255), 1
        )  # 畫出 y 座標的水平線
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

# 顯示結果
cv2.imshow("Measurement Results", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 儲存結果圖片
cv2.imwrite("xxx.png", output_image)

