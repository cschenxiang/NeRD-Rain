<div align="center">

# 【CVPR'2024🔥】Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining
</div>

> Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining
>
> [Xiang Chen](https://cschenxiang.github.io/), [Jinshan Pan](https://jspan.github.io/), [Jiangxin Dong](https://scholar.google.com/citations?user=ruebFVEAAAAJ&hl=en&oi=ao)
>
> Nanjing University of Science and Technology
>
> Primary contact: Xiang Chen (chenxiang@njust.edu.cn)

## 📣 News
- [24-02-27] Our paper has been accepted to CVPR 2024.

## 📌 Overview
![avatar](figs/Overview.jpg)

## 🔑 Setup
Type the command:
```
pip install -r requirements.txt
```

Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

## 🧩 Dataset Preparation
| Datasets | Download Link | 
|:-----: |:-----: |
| Rain200L | [Baidu Netdisk](https://pan.baidu.com/s/1rTb4qU3fCEA4MRpQss__DA?pwd=s2yx) (s2yx) |
| Rain200H | [Baidu Netdisk](https://pan.baidu.com/s/1KK8R2bPKgcOX8gMXSuKtCQ?pwd=z9br) (z9br) |
| DID-Data | [Baidu Netdisk](https://pan.baidu.com/s/1aPFJExxxTBOzJjngMAOQDA?pwd=5luo) (5luo) |
| DDN-Data | [Baidu Netdisk](https://pan.baidu.com/s/1g_m7RfSUJUtknlWugO1nrw?pwd=ldzo) (ldzo) |
| SPA-Data | [Baidu Netdisk](https://pan.baidu.com/s/1YfxC5OvgYcQCffEttFz8Kg?pwd=yjow) (yjow) |

## 🚨 Performance Evaluation
See folder "evaluations" 

1) *for Rain200L/H and SPA-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/swz30/Restormer/blob/main/Deraining/evaluate_PSNR_SSIM.m).

2) *for DID-Data and DDN-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/hongwang01/RCDNet/tree/master/Performance_evaluation).

<img src = "figs/table.png">

## 🚀 Visual Deraining Results
| Datasets | DualGCN | SPDNet | Uformer | Restormer |
|:-----: |:-----: |:-----: |:-----: |:-----: |
| Rain200L | [Baidu Netdisk](https://pan.baidu.com/s/1o9eLMv7Zfk_GC9F4eWC2kw?pwd=v8qy) (v8qy) | [Baidu Netdisk](https://pan.baidu.com/s/1u9F4IxA8GCxKGk6__W81Og?pwd=y39h) (y39h) | [Baidu Netdisk]() (N/A) |[Baidu Netdisk](https://pan.baidu.com/s/1jv6PUMO7h_Tc4ovrCLQsSw?pwd=6a2z) (6a2z) |
| Rain200H | [Baidu Netdisk](https://pan.baidu.com/s/1QiKh5fTV-QSdnwMsZdDe9Q?pwd=jnc9) (jnc9) | [Baidu Netdisk](https://pan.baidu.com/s/1wSTwW6ewBUgNLj7l7i6HzQ?pwd=mry2) (mry2) | [Baidu Netdisk]() (N/A) |[Baidu Netdisk](https://pan.baidu.com/s/16R0YamX-mfn6j9sYP7QpvA?pwd=9m1r) (9m1r) |
| DID-Data | [Baidu Netdisk](https://pan.baidu.com/s/1Wh7eJdOwXPABz5aOBPDHaA?pwd=3gdx) (3gdx) | [Baidu Netdisk](https://pan.baidu.com/s/1z3b60LHOyi8MLcn8fdNc8A?pwd=klci) (klci) | [Baidu Netdisk](https://pan.baidu.com/s/1fWLjSCSaewz1QXdddkpkIw?pwd=4uur) (4uur) |[Baidu Netdisk](https://pan.baidu.com/s/1b8lrKE82wgM8RiYaMI6ZQA?pwd=1hql) (1hql) |
| DDN-Data | [Baidu Netdisk](https://pan.baidu.com/s/1ML1A1boxwX38TGccTzr6KA?pwd=1mdx) (1mdx) | [Baidu Netdisk](https://pan.baidu.com/s/130e74ISgZtlaw8w6ZzJgvQ?pwd=19bm) (19bm) | [Baidu Netdisk](https://pan.baidu.com/s/1cWY7piDJRF05qKYPNXt_cA?pwd=39bj) (39bj) |[Baidu Netdisk](https://pan.baidu.com/s/1GGqsfUOdoxod9vAUxB54PA?pwd=crj4) (crj4) |
| SPA-Data | [Baidu Netdisk](https://pan.baidu.com/s/16RHVyrBoPnOhW1QuglRmlw?pwd=lkeb) (lkeb) | [Baidu Netdisk](https://pan.baidu.com/s/1J0ybwnuT__ZGQZNbMTfw8Q?pwd=dd98) (dd98) | [Baidu Netdisk]() (N/A) |[Baidu Netdisk](https://pan.baidu.com/s/1IG4T1Bz--FrDAuV6o-fykA?pwd=b40z) (b40z) |

| Datasets | IDT | DRSformer | NeRD-Rain-S | NeRD-Rain |
|:-----: |:-----: |:-----: |:-----: |:-----: |
| Rain200L | [Baidu Netdisk](https://pan.baidu.com/s/1jhHCHT64aDknc4g0ELZJGA?pwd=v4yd) (v4yd) | [Baidu Netdisk](https://pan.baidu.com/s/1-ElpyJigVnpt5xDFE6Pqqw?pwd=hyuv) (hyuv) | [Baidu Netdisk]() (N/A) |[Baidu Netdisk]() (N/A) | 
| Rain200H | [Baidu Netdisk](https://pan.baidu.com/s/10TZzZH0HisPV0Mw-E4SlTQ?pwd=77i4) (77i4) | [Baidu Netdisk](https://pan.baidu.com/s/13aJKxH7V_6CIAynbkHXIyQ?pwd=px2j) (px2j) | [Baidu Netdisk]() (N/A) |[Baidu Netdisk]() (N/A) |
| DID-Data | [Baidu Netdisk](https://pan.baidu.com/s/1svMZAUvs6P6RRNGyCTaeAA?pwd=8uxx) (8uxx) | [Baidu Netdisk](https://pan.baidu.com/s/1Xl3q05rZYmNEtQp5eLTTKw?pwd=t879) (t879) | [Baidu Netdisk]() (N/A) |[Baidu Netdisk]() (N/A) | 
| DDN-Data | [Baidu Netdisk](https://pan.baidu.com/s/1FSf3-9HEIQ-lLGRWesyszQ?pwd=0ey6) (0ey6) | [Baidu Netdisk](https://pan.baidu.com/s/1D36Z0cEVPPbm5NljV-8yoA?pwd=9vtz) (9vtz) | [Baidu Netdisk]() (N/A) |[Baidu Netdisk]() (N/A) |
| SPA-Data | [Baidu Netdisk](https://pan.baidu.com/s/16hfo5VeUhzu6NYdcgf7-bg?pwd=b862) (b862) | [Baidu Netdisk](https://pan.baidu.com/s/1Rc36xXlfaIyx3s2gqUg_Bg?pwd=bl4n) (bl4n) | [Baidu Netdisk]() (N/A) |[Baidu Netdisk]() (N/A) |


## 👍 Acknowledgement
Thanks for their awesome works ([DeepRFT](https://github.com/INVOKERer/DeepRFT) and [NeRCo](https://github.com/Ysz2022/NeRCo)).

## 📘 Citation
Please consider citing our work as follows if it is helpful.
```
@InProceedings{NeRD-Rain,
    author={Chen, Xiang and Pan, Jinshan and Dong, Jiangxin}, 
    title={Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2024}
}
```

