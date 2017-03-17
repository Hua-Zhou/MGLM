---
layout: default
title: MGLM
---

## MGLM Toolbox for Matlab

MGLM toolbox is a collection of Matlab functions for multi-response GLM regression and sparse regression.

The toolbox is developed by [Hua Zhou](http://hua-zhou.github.io) and [Yiwen Zhang](http://Yiwen-Zhang.github.io).

### Compatibility

The code is tested on Matlab R2017a, but should work on other versions of Matlab with no or little changes. Current version works on these platforms: Windows 64-bit, Linux 64-bit, and Mac (Intel 64-bit). Type `computer` in Matlab's command window to determine the platform.

### Installation (Matlab version >= 2014b)

Download the Matlab toolbox installation file [MGLM.mltbx](./MGLM.mltbx). Double click the downloaded file and you should be good to go. If for some reasons, it does not work. Follow the below instructions for Matlab version < 2014b.


### Installation (Matlab version < 2014b)

1. Download `ZIP File` file using the links on the left.  2. Extract the zip file.  
```
unzip Hua-Zhou-MGLM-xxxxxxx.zip
```
3. Rename the folder from *Hua-Zhou-MGLM-xxxxxxx* to *MGLM*.  
```
mv Hua-Zhou-MGLM-xxxxxxx MGLM
```
4. Add the *MGLM* folder to Matlab search path. Start Matlab, cd to the *MGLM* directory, and execute the following commands  
`addpath(pwd)	 %<-- Add the toolbox to the Matlab path`  
`save path	 %<-- Save for future Matlab sessions`
5. Go through following tutorials for the usage. For help of individual functions, type `?` followed by the function name in Matlab.

### Tutorial

* [Dirichlet-Multinomial distribution](./html/demo_dirmn.html)
* [Generalized Dirichlet-Multinomial distribution](./html/demo_gendirmn.html) 
* [Negative multinomial distribution](./html/demo_negmn.html) 
* [Multinomial-logit regression and sparse regression](./html/demo_mnlogitreg.html)
* [Dirichlet-Multinomial regression and sparse regression](./html/demo_dirmnreg.html)
* [Generalized Dirichlet-Multinomial regression and sparse regression](./html/demo_gendirmnreg.html)
* [Negative multinomial regression and sparse regression](./html/demo_negmnreg.html)


### How to cite

If you use this toolbox in any way, please cite the software itself along with at least one publication or preprint.

* Software reference  
H Zhou and Y Zhang. Matlab MGLM Toolbox Version 1.0.0, Available online, March 2017.  
* Default article to cite for the toolbox  
Y Zhang, H Zhou, J Zhou, and W Sun. (2017) Regression models for multivariate count data. [_Journal of Computational and Graphical Statistics_](http://www.tandfonline.com/doi/abs/10.1080/10618600.2016.1154063), 26(1):1-13. \[[pdf](http://hua-zhou.github.io/media/pdf/ZhangZhouZhouSun17mglm.pdf)\]

### Contacts

Hua Zhou <huazhou@ucla.edu>. 

