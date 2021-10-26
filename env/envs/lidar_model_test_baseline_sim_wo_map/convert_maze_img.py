from PIL import Image
import numpy as np

img = Image.open('/home/awesomericky/intern/raisimLib/raisimGymTorch/heighmap/maze3.png')
gray_img = img.convert('L')
width, height = gray_img.size
gray_img = np.array(gray_img.getdata())
gray_img = np.where(gray_img > 0, 0, 255)
gray_img = gray_img.reshape((height, width)).astype(np.uint8)
gray_img = Image.fromarray(gray_img)
gray_img.save('/home/awesomericky/intern/raisimLib/raisimGymTorch/heighmap/maze4.png')


