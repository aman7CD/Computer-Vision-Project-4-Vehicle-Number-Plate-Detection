
import cv2
nplate_cascade = cv2.CascadeClassifier("C:\\Users\\Aman\\machine learning projects\\MACHINE LEARNING\\PycharmProjects\\opencv_projects\\haarcascades\\haarcascades\\haarcascade_russian_plate_number.xml")
capt = cv2.VideoCapture(0)
capt.set(3,640)
capt.set(4,480)
capt.set(10,150)
count = 0
while True:
    success,img = capt.read()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numplate = nplate_cascade.detectMultiScale(imggray, 1.1, 10)
    for (x,y,w,h) in numplate:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,"Number Plate",(x,y-4),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
        img_cropped = img[y:y+h,x:x+w]
        cv2.imshow("numb_plate", img_cropped)
        cv2.imshow("result", img)


    if cv2.waitKey(1) & 0xff == ord("s"):
        cv2.imwrite("C:\\Users\\Aman\\machine learning projects\\MACHINE LEARNING\\PycharmProjects\\opencv_projects\\Number Plate Detection\\scanned_plates\\number_plate_"+str(count)+".jpg", img_cropped)
        cv2.rectangle(img,(0,200),(600,300),(150,0,0),cv2.FILLED)
        cv2.putText(img,"Saved",(150,265),cv2.FONT_HERSHEY_TRIPLEX,2,(0,255,0),3)
        cv2.imshow("result",img)
        cv2.waitKey(500)
        count = count+1
