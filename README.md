# Time measurments for applying median filter to image with salt and pepper noise
Average for 5 measurments of each size
| Image size, pixels | Time CPU, msec | Time GPU, msec | CPU/GPU ratio|
|-------------|----------|----------|------------|
|128| 32| 1.16| 27.6|
|256 | 152 | 5.6 | 27.1|
|512| 556 | 14 | 39.7|
|1024| 1645 | 55 | 29.9 |
|2048| 6715 | 216 |31.1|

# Source 
<img src="https://github.com/Opsy1169/salt_and_pepper/blob/master/data/eltsin_noise.bmp?raw=true"  width="512">

# CPU result 
<img src="https://github.com/Opsy1169/salt_and_pepper/blob/master/data/resultCPU.bmp?raw=true"  width="512">

# GPU result 
<img src="https://github.com/Opsy1169/salt_and_pepper/blob/master/data/resultGPU.bmp?raw=true"  width="512">

