import cv2
import numpy as np
import matplotlib.pyplot as plt
from patchmatch import *
import torch

ava_img = cv2.imread("ava.png")
mona_img = cv2.imread("mona.png")

ava_img=cv2.resize(ava_img,(224,224))
mona_img = cv2.resize(mona_img,(224,224))

ava_img=(ava_img/255).astype(np.float32)
mona_img=(mona_img/255).astype(np.float32)

ava_img=torch.tensor(ava_img).permute(2,0,1).unsqueeze(0)
mona_img=torch.tensor(mona_img).permute(2,0,1).unsqueeze(0)

pm=PatchMatch(ava_img,mona_img,ava_img,mona_img,3)
nnf= pm.run(num_iters=10, allow_diagonals=True)
plt.imshow(cv2.cvtColor(pm.reconstruct_avg(patch_size=1), cv2.COLOR_BGR2RGB))
plt.show()
plt.imshow(cv2.cvtColor(cv2.imread("mona.png"), cv2.COLOR_BGR2RGB))

plt.show()
plt.imshow(cv2.cvtColor(pm.visualize(), cv2.COLOR_BGR2RGB))