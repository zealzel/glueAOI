import cv2
import numpy as np

# 讀取圖片
# image_path = "images/GL135141V1L0XFCA1+P13434443114-Normal.jpg"
image_path = "images/GL135141V1L0XFCA1+P13434443114-Normal.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 使用 Gaussian 模糊來平滑圖像，減少雜訊
blur_radius = 13
image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

# 使用自適應閾值來進行二值化
binary = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
    # blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
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

# 計算指定位置的寬度
def measure_width(mask, y):
    x_indices = np.where(mask[y, :] == 255)[0]
    if len(x_indices) >= 2:
        width = x_indices[-1] - x_indices[0]
        return width
    return None

p1,radius = (300,850), 10
p2,radius = (300,900), 10
cv2.circle(mask, p1, radius, 255, -1)
cv2.circle(mask, p2, radius, 255, -1)


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
cv2.imshow("Smoothed Contour", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

