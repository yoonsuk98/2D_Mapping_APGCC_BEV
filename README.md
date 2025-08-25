# 2D Mapping (APGCC + BEV)

This project is conducted at [PIASPACE](https://www.pia.space/) internship.

## Setup
1) Create a conda environment and activate it.
    ```
    conda create --name 2dmapping python=3.8 -y
    conda activatre 2dmapping
    ```
2) Clone and enter into repo directory.
    ```
    git clone "this repo"
    ```
3) Install torch
    ```
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    ```
4) Install remaining dependencies
    ```
    pip install -r requirements.txt
    ```
5) Download pretrained APGCC checkpoints and place them into path (./apgcc/outputs/).
    - [SHHA APGCC checkpoint](https://drive.google.com/file/d/1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu/view?usp=sharing)
    - You can also complete by the command:
    ```
    cd apgcc
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu' -O ./output/SHHA_best.pth
    ```

6) Prepare Wildtrack Dataset
 Please see this [page](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) to download the [Wildtrack_dataset_full.zip](http://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip) into path (./apgcc)

## execute 2D Mapping
   After Projection, we use DBSCAN, NMS. 
1) DBSCAN
   ```
   cd apgcc
   python APGCC_DBSCAN.py
   ```

2) NMS
    ```
    cd apgcc
    python APGCC_NMS.py
    ```

3) NMS + SORT
    ```
    cd apgcc
    python APGCC_NMS_SORT.py
    ```

## Make video
1) APGCC RESULT
    ```
    cd apgcc
    python APGCC_result_make_video.py
    ```
2) BEV Result
    ```
    cd apgcc
    python BEV_make_video.py
    ```
3) TRACKING Result
    ```
    cd apgcc
    python Tracking_make_video.py
    ```


## Reference

```python
@inproceedings{chen2024improving,
	title={Improving point-based crowd counting and localization based on auxiliary point guidance},
	author={Chen, I-Hsiang and Chen, Wei-Ting and Liu, Yu-Wei and Yang, Ming-Hsuan and Kuo, Sy-Yen},
	booktitle={European Conference on Computer Vision},
	pages={428--444},
	year={2024},
	organization={Springer}
  }
```
```python
@inproceedings{MVOT24,
title={Mahalanobis Distance-based Multi-view Optimal Transport for Multi-view Crowd Localization},
author={Qi Zhang and Kaiyi Zhang and Antoni B. Chan and Hui Huang},
booktitle={European Conference on Computer Vision},
pages={19--36},
year={2024},
}
```


