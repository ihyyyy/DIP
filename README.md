# Course Materials for DIP (Digital Image Processing)

## requirement 要求

## Assignment 04 

### 代码
```

#compute jacobian
J_proj[:, 0, 0] = K[0,0]/cam_points[:,2]  # dx/dX
J_proj[:, 0, 1] = 0  # dx/dY
J_proj[:, 0, 2] = -(K[0,0]*cam_points[:,0])/(cam_points[:,2]*cam_points[:,2])  # dx/dZ
J_proj[:, 1, 0] = 0  # dy/dX
J_proj[:, 1, 1] = K[1,1] / cam_points[:,2]  # dy/dY
J_proj[:, 1, 2] = -(K[1,1]*cam_points[:,1]) / (cam_points[:,2]*cam_points[:,2])   # dy/dZ



# compute gaussian value
det_covs = torch.det(covs2D)  # (N,)
inv_covs = torch.linalg.inv(covs2D)  # (N, 2, 2)

# Compute the Mahalanobis distance (N, H, W)
dx_cov_inv = torch.einsum('nhwi,nij->nhwj', dx, inv_covs)  # (N, H, W, 2)

mahalanobis = torch.sum(dx_cov_inv * dx, dim=-1)  # (N, H, W)

# Compute Gaussian values (N, H, W)
gaussian = (1. / (2 * torch.pi * det_covs.sqrt())).unsqueeze(-1).unsqueeze(-1) * torch.exp(-0.5 * mahalanobis)  # (N, H, W)


#alpha composition
T = torch.cumprod(torch.cat([torch.ones(1, self.H, self.W, device=alphas.device), 1 - alphas[:-1]], dim=0), dim=0)
weights = alphas * T
```
### result 结果
![epoch 70](./Assignments/04_3DGS/data/chair/checkpoints/debug_images/epoch_0070/r_2.png)


## Assignment 03  

### 运行
![](./Assignments/03_PlayWithGANs/02/ycz.jpg)
```
python invert.py
```
![](./Assignments/03_PlayWithGANs/02/project.png)
```
python landmark_dect.py
```

```
make_it_smile.py
```

![](./Assignments/03_PlayWithGANs/02/smile_face.jpg)


### result 结果



## Assignment 02 

### result 结果



## Assignment 01 


### result 结果





### [上课课件（持续更新）](https://rec.ustc.edu.cn/share/705bfa50-6e53-11ef-b955-bb76c0fede49) 


### 编程入门资料
不用花很多时间专门学习，在课程作业或科研项目过程中一边学一边善用搜索即可
- [Python 入门](https://github.com/walter201230/Python)
- [OpenCV Docs](https://codec.wang/docs/opencv)
- [PyTorch 入门](https://github.com/datawhalechina/thorough-pytorch)

### 课程作业
- 作业会更新在[Assignments](Assignments/)文件夹
- [作业提交模板](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md)
- [Assignment_01](Assignments/01_ImageWarping) (Due: 2024.10.01)
- [Assignment_02](Assignments/02_DIPwithPyTorch/) (Due: 2024.10.31)
- [Assignment_03](Assignments/03_PlayWithGANs/) (Due: 2024.11.30)
- [Assignment_04](Assignments/04_3DGS/) (Due: 2025.01.06)
