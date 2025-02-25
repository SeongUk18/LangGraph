from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# PaddleOCR 초기화 (한국어 지원, 기울기 보정 포함)
ocr = PaddleOCR(use_angle_cls=True, lang="korean")

# OCR할 이미지 경로
img_path = "data/test.png"

# 이미지에 대해 OCR 수행 (cls=True로 텍스트 각도 분류 포함)
result = ocr.ocr(img_path, cls=True)

# OCR 결과 존재 여부 확인
if not result or not result[0]:
    raise ValueError("OCR 결과가 없습니다.")

# 전체 OCR 결과 출력
print("Full OCR result:")
for idx, line in enumerate(result[0]):
    print(f"Line {idx}: {line}")

# OCR 결과에서 텍스트만 추출하여 출력
print("\nExtracted texts:")
valid_results = []
for idx, line in enumerate(result[0]):  # OCR 결과 리스트에서 순회
    try:
        box = line[0]  # 바운딩 박스 좌표
        text = line[1][0]  # OCR 인식된 텍스트
        print(f"Line {idx} text: {text}")  # 텍스트 출력

        valid_results.append((box, text))

    except Exception as e:
        print(f"Skipping invalid line {idx}: {e}")

# OpenCV를 이용하여 이미지 읽기
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"이미지를 불러올 수 없습니다: {img_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Windows의 경우 한글 폰트 지정
font_path = "C:\\Windows\\Fonts\\malgun.ttf"

# `draw_ocr()`에 맞는 형식으로 전달 (box, text 쌍 리스트)
if valid_results:
    try:
        image_with_boxes = draw_ocr(
            img,
            [x[0] for x in valid_results],
            [x[1] for x in valid_results],
            font_path=font_path,
        )

        # 결과 시각화
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_boxes)
        plt.axis("off")
        plt.title("OCR")
        plt.show()
        # plt.savefig("Test.png")
    except Exception as e:
        print(f"draw_ocr()에서 오류 발생: {e}")
else:
    print("No valid OCR results found for visualization.")
