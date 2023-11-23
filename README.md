## Hyperspectral Image Classification via Cross-Domain Few-Shot Learning With Kernel Triplet Loss

This is a code demo for the folowing paper:

K. -K. Huang, H. -T. Yuan, C. -X. Ren, Y. -E. Hou, J. -L. Duan and Z. Yang, 
"Hyperspectral Image Classification via Cross-Domain Few-Shot Learning With Kernel Triplet Loss," 
in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-18, 2023. 
Art no. 5530818, https://doi.org/10.1109/TGRS.2023.3332051 


## Requirements
CUDA = 10.2
Python = 3.6
Pytorch = 1.6


## dataset

You can download the EMAP feature for hyperspectral datasets in mat format at: https://pan.baidu.com/s/1j9fNXQLMfJO18b0JP2VpCQ?pwd=0561, and move the files to `./data` folder.

or you can extract EMAP feature by the matlab source code 'extractEMAP.rar'. 


This matlab source code to extract EMAP is provided by the following paper:
[1] L. He, J. Li, C. Liu and S. Li, 
Recent Advances on Spectral-Spatial Hyperspectral Image Classification: An Overview and New Guidelines. 
IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 3, pp. 1579-1597, March 2018.

The EMAP features are built by:
[2] M. D. Mura, A. Villa, J. A. Benediktsson, J. Chanussot, and L. Bruzzone, 
Classification of hyperspectral images by using extended morphological attribute profiles and independent component analysis,
IEEE Geoscience and Remote Sensing Letters, vol. 8, no. 3, pp. 542-546, May 2011.


## Usage:

Change data_path if necessery.
Run main_CFSLKT.py

