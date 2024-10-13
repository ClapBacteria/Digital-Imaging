import cv2

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("C:/opencv/upscaling/EDSR_x4.pb")   # 미리 학습된 딥러닝 모델 파일 경로
sr.setModel("edsr", 4)  
# 모델 유형과 업스케일 비율 설정 esdr은 2배, 3배, 4배의 초해상도 업스케일링 가능

# 이미지 읽기
img = cv2.imread('C:/opencv/upscaling/1.jpg')

#이미지를 읽지 못했을때 출력
if img is None:
    print("이미지를 불러오지 못했습니다. 파일 경로를 확인하세요.")
    exit()

# 이미지 초해상도 업스케일링 실행
result = sr.upsample(img)

# 결과 이미지 저장
save_img = 'C:/opencv/upscaling/sr_img.jpg'
cv2.imwrite(save_img, result)

print(f"화질이 개선된 이미지가 {save_img}에 저장되었습니다.")

