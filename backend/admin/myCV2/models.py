from django.db import models
import cv2

from admin.common.models import ValueObject, Reader


class MyCV2(object):
    def __init__(self):
        self.vo = ValueObject()
        self.reader = Reader()
        self.vo.context = 'admin/MyCV2/data/'

    def face_detect(self):
        vo = self.vo
        reader = self.reader
        vo.fname = 'haarcascade_frontalface_alt.xml'
        face_filter = reader.new_file(vo)
        vo.fname = 'girl2.jpg'
        image = cv2.imread(reader.new_file(vo))
        cascade = cv2.CascadeClassifier(face_filter)
        face = cascade.detectMultiScale(image, minSize=(150, 150))
        if len(face) == 0:
            print('얼굴 인식 실패')
            quit()
        for(x, y, w, h) in face:
            red = (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), red, thickness=20)

        cv2.imwrite(f'{vo.context}face_detection.png', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def lena(self):
        vo = self.vo
        reader = self.reader
        vo.fname = 'lena.jpg'
        lena = reader.new_file(vo)
        original = cv2.imread(lena, cv2.IMREAD_COLOR)
        gray = cv2.imread(lena, cv2.IMREAD_GRAYSCALE)
        unchanged = cv2.imread(lena, cv2.IMREAD_UNCHANGED)

        cv2.imwrite(f'{vo.context}lena_Original.png', original)
        cv2.imwrite(f'{vo.context}lena_Gray.png', gray)
        cv2.imwrite(f'{vo.context}lena_Unchanged.png', unchanged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def girl(self):
        vo = self.vo
        reader = self.reader
        vo.fname = 'girl.jpg'
        girl = reader.new_file(vo)
        original = cv2.imread(girl, cv2.IMREAD_COLOR)
        negative = 255 - original
        cv2.imwrite(f'{vo.context}girl_negative.png', negative)
        bgr2gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        th = 90
        bgr2gray[bgr2gray > th] = 255
        bgr2gray[bgr2gray < th] = 0
        cv2.imwrite(f'{vo.context}girl_cvtColor.png', bgr2gray)
        small_image = original[150:450, 150:450]
        small_image = cv2.resize(small_image, (400,400))
        cv2.imwrite(f'{vo.context}girl_small_image.png', small_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Create your models here.
