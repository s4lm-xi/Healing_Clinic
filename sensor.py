import time
import cv2

#find path of xml file containing haarcascade file
cascPathface = 'python/xml/haarcascade_frontalface_default.xml'
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file

index = 0
cap = cv2.VideoCapture(0)
start = time.time()
while True:
    _, image = cap.read()
    if _:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #convert image to Greyscale for haarcascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, 'Salman', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, (0, 255, 0), 2)

        cv2.imshow('Results', image)
        if cv2.waitKey(1) and float(time.time()-start) >= 10.0:
            cv2.destroyAllWindows()
            cap.release()
            False
            break

print("hahaha")
