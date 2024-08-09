import cv2
import numpy as np

# 讀取圖片
# image_path = "/mnt/data/image.png"
image_path = "images/GL135141V1L0XFCA1+P13434443114-Normal.jpg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 應用二值化處理，分離白色區域
_, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# 尋找輪廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的輪廓
largest_contour = max(contours, key=cv2.contourArea)

# 將輪廓繪製在黑色背景上
mask = np.zeros_like(image)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)


# 計算指定位置的寬度
def measure_width(mask, y):
    x_indices = np.where(mask[y, :] == 255)[0]
    if len(x_indices) >= 2:
        width = x_indices[-1] - x_indices[0]
        return width
    return None


# 指定量測位置的Y座標
y_positions = [61, 163, 252]

# 量測寬度
for y in y_positions:
    width = measure_width(mask, y)
    if width:
        print(f"位置 Y = {y} 的寬度為: {width} 像素")
    else:
        print(f"無法在位置 Y = {y} 找到有效的寬度")

# 顯示結果
cv2.imshow("Detected Contour", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
