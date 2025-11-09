import pywt
import cv2
import numpy as np


def extract_image_features(image_path, wavelet='haar'):
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 2. 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        # 3. 进行二维离散小波变换
        coeffs = pywt.dwt2(gray_image, wavelet)
        LL, (LH, HL, HH) = coeffs

        # 4. 提取特征
        features = []
        # 提取近似子带（LL）的特征
        LL_mean = np.mean(LL)
        LL_var = np.var(LL)
        features.extend([LL_mean, LL_var])

        # 提取水平细节子带（LH）的特征
        LH_mean = np.mean(LH)
        LH_var = np.var(LH)
        features.extend([LH_mean, LH_var])

        # 提取垂直细节子带（HL）的特征
        HL_mean = np.mean(HL)
        HL_var = np.var(HL)
        features.extend([HL_mean, HL_var])

        # 提取对角细节子带（HH）的特征
        HH_mean = np.mean(HH)
        HH_var = np.var(HH)
        features.extend([HH_mean, HH_var])

        return features
    except Exception as e:
        print(f"小波变换过程中出现错误: {e}")
        return None


# 示例使用
image_path = 'n01440764_105.JPEG'
features = extract_image_features(image_path)
if features:
    print("提取的图像特征:", features)
    