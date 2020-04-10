import cv2
import matplotlib.pyplot as plt
from ISR.models import RDN
import time
import sys
import os

t = time.time()
rdn = RDN(weights='noise-cancel')
print("Loaded model in %.3f s." % (time.time() - t))

image = cv2.imread('test1.jpg')
image = image[:, :, ::-1]

low_res_cropped = image[633: 662, 575: 645, :]

t = time.time()
sr_image = rdn.predict(low_res_cropped)
print("Image processed in %.3f s." % (time.time() - t))

fig, axarr = plt.subplots(1, 3)
fig.suptitle("Super resolucion usando la red neuronal ESRGAN (Wang et al. 2018)", fontsize=16)

axarr[0].set_title("Imagen original")
axarr[0].axis('off')
axarr[0].imshow(image)

axarr[1].set_title("Seccion recortada")
axarr[1].axis('off')
axarr[1].imshow(low_res_cropped)

axarr[2].set_title("Conversion a alta resolucion")
axarr[2].axis('off')
axarr[2].imshow(sr_image)

fig.subplots_adjust(hspace=0)
plt.show()
