import cv2
import numpy as np
import time
from datetime import datetime

SEGMENTS_DICT = {
    (0,255,0,255,0,0,255,0,255,0,0,255,0,255,0) : "0",
    (0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0) : "1",
    (0,255,0,0,255,0,255,0,0,255,0) :"4",
    (0,255,0,255,0,255,0,0,255,0,0,255,0,255,0): "6",
    (0,255,0,0,255,0,0,255,0): "7",
    (0,255,0,255,0,255,0,0,255,0,255,0,0,255,0,255,0): "8",
    (0,255,0,255,0,255,0,0,255,0,255,0,0,255,0):"9"}

DICT_2_3_5 = {
    (0,0,255,0): "2",
    (0,0): "3",
    (0,255,0,0): "5"
}    

def get_segments(crop):    
    row, colum = crop.shape
    mid_colum = crop[:, colum // 2]
    q1 = crop[row//4, :]
    q4 = crop[row*3//4, :]
    mid_colum_segments = mid_colum[np.insert(np.diff(mid_colum) != 0, 0, True)]
    q1_segments = q1[np.insert(np.diff(q1) != 0, 0, True)]
    q4_segments = q4[np.insert(np.diff(q4) != 0, 0, True)]
    
    return tuple(int(i) for i in np.concatenate([mid_colum_segments, q1_segments, q4_segments]))

def _2_3_5(crop):
    row, colum = crop.shape
    mid_colum = crop[:, colum // 2]
    q1 = crop[row//4, : colum // 2]
    q4 = crop[row*3//4, : colum // 2]
    q1_segments = q1[np.insert(np.diff(q1) != 0, 0, True)]
    q4_segments = q4[np.insert(np.diff(q4) != 0, 0, True)]

    return tuple(int(i) for i in np.concatenate([q1_segments, q4_segments]))


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
batch_size = 50
data = []
txt_path = f"./{datetime.now().strftime('%H_%M')}.txt"

while True:
    time.sleep(5)
    ret,frame = cap.read()
    if ret:
        im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (9,9), 0)
        dst = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        thr = dst.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
        dst = cv2.dilate(dst, np.ones((15,15)), iterations= 1)

        contours,_ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_cont = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * im_gray.shape[0])
    
        segments_digits = []
        for cnt in sorted_cont:
            x, y, w, h = cv2.boundingRect(cnt)
            area = h*w
            if area > 1000 and h > 120:
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
                crop = thr[y:y+h, x:x+w]
                segments = get_segments(crop)

                try:
                    if segments == (0, 255, 0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0):
                        other_segments = _2_3_5(crop)
                        segments_digits.append(DICT_2_3_5[other_segments])
                    else:
                        segments_digits.append(SEGMENTS_DICT[segments])
                except:
                    continue


        V,A = segments_digits[:len(segments_digits)//2], segments_digits[len(segments_digits)//2:]
        data.append(f"{datetime.now().strftime('%H:%M:%S')} : V = {''.join(V)}, A = {''.join(A)}")
    
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if len(data) == batch_size:
        print("flush")
        with open(txt_path, "a") as f:
            f.write("\n".join(data) + "\n")
        data.clear()

cap.release()
cv2.destroyAllWindows()

