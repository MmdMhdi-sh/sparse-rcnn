| Model | Params | Test mAP@[.5:.95] ↑ | mAP@.5 ↑ | mAP@.75 ↑ | AR@100 ↑ | AP circle / square / triangle | Exact object count ↑ | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse R-CNN (FPN P2-P4, 3 stages) | 6.00M | 0.762 ± 0.020 | 0.898 ± 0.012 | 0.832 ± 0.020 | 0.846 ± 0.013 | 0.831 / 0.701 / 0.754 | 0.63 ± 0.03 | 77, 76, 76 |
| Single 16x16 feature map (no FPN) | 5.68M | 0.680 ± 0.011 | 0.887 ± 0.003 | 0.768 ± 0.013 | 0.773 ± 0.010 | 0.770 / 0.598 / 0.670 | 0.64 ± 0.03 | 31, 47, 56 |

Mean ± std over seeds [42, 43, 44]. Test set: 1,000 freshly generated images (2,020 objects), never used for checkpoint selection (that uses the 100-image validation split). mAP is COCO-style over all 20 detections per image. 'Exact object count' is the share of test images with exactly as many detections above score 0.3 as objects. Hardware: Tesla T4.
