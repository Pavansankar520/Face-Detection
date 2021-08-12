import cv2
import matplotlib.pyplot as plt
 
facial_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
 
images = cv2.imread("group.jpg") #image path
images = cv2.resize(images,(800,500)) #resizing the image
gray_scale = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY) #converting brg to gray scale

facials = facial_cascade.detectMultiScale(gray_scale, scaleFactor =1.05, minNeighbors=3)
print(facials)
for x,y,w,h in facials:
    images = cv2.rectangle(images, (x,y), (x+w,y+h),(0,255,0),3)

cv2.imshow("facialdetection", images)
plt.imshow(images)
plt.show()

#clearing ram memory 
cv2.waitKey(0)
 
cv2.destroyAllWindows()
