import cv2

capture = cv2.VideoCapture(1)
i = 0
while True:
    ret, image = capture.read()
    cv2.imshow("image", image)
    q = cv2.waitKey(1)
    if q == ord('u'):
        break
    if q == ord('w'):
        cv2.imwrite(".\\trueData\\cigarette\\" + str(i) + ".jpg", image)
        i += 1
    print(i)