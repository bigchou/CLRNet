from glob import glob
import cv2, os
for item in glob('images/*.jpg'):
    img = cv2.imread(item)
    img = cv2.resize(img, (1640, 590))
    print(img.shape)
    cv2.imwrite('images1640590/%s'%(os.path.basename(item)), img)
