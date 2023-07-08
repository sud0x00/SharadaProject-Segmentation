The following folder contains notebooks which are being tested as of now. 

The the pipeline struggles to perform character level segmentation , hence various methods will be tests and the outputs will be mentioned in this readme file. 




![image](https://github.com/sud0x00/SharadaProject-Segmentation/assets/91898207/11fa9990-909b-426e-8b42-a17376d8e1a4)


```
import cv2

image = cv2.imread("/content/Line_Segment/line_5_vbV6I6xO3D042pe.jpg")
image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

res, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # threshold
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

dilated = cv2.dilate(thresh, kernel, iterations=5)

contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

coord = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if h > 300 and w > 300:
        continue
    if h < 40 or w < 40:
        continue
    coord.append((x, y, w, h))

coord.sort(key=lambda tup: tup[0])  # if the image has only one sentence sort in one axis

count = 0
for cor in coord:
    x, y, w, h = cor
    t = image[y: y + h, x: x + w, :]
    cv2.imwrite(str(count) + ".png", t)
    count += 1

print("number of char in image:", count)
```

The above code performed word level segmentation and the character level segmentation was not accomplished due to the lack of spaces between characters , tweaking the program leads to false segmentations.



Tesseract Output: 

Here , using the following code pytesseract was used to apply character segmentation but the outputs weren't accuracte since the model wasn't trained on the manuscripts and the default english model was used.

![image](https://github.com/sud0x00/SharadaProject-Segmentation/assets/91898207/24a1838c-0c14-4878-b54c-8bc386ebea7e)
```
W 10 18 65 60 0
L 64 17 158 76 0
H 144 15 185 57 0
Z 185 4 228 59 0
A 221 1 261 76 0
S 258 19 304 62 0
A 305 15 340 62 0
I 333 13 860 76 0
A 404 0 469 76 0
N 461 0 513 76 0
I 487 0 538 76 0
D 519 0 570 76 0
V 569 1 646 76 0
I 620 1 671 76 0
L 645 1 697 76 0
T 670 1 735 76 0
A 702 1 766 76 0
I 740 1 792 76 0
E 765 1 817 76 0
A 797 1 860 76 0
```
```
!sudo apt install tesseract-ocr
!pip install pytesseract

import cv2
import pytesseract
from google.colab.patches import cv2_imshow

file = '/content/Line_Segment/line_5_vbV6I6xO3D042pe.jpg'

img = cv2.imread(file)
h, w, _ = img.shape

boxes = pytesseract.image_to_boxes(img)

for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2_imshow(img)
print(boxes)
```
Target Segmentation :

![image](https://github.com/sud0x00/SharadaProject-Segmentation/assets/91898207/725d6a0f-0e43-4965-b6ff-b440cd5c96dc)

