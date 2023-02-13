
23/02/13  
===  
update tensorflow/pytorch test log  
**tensorflow image**: l4t-tensorflow:r32.5.0-tf2.3-py3  
**pytorch image**: l4t-pytorch:r32.5.0-pth1.7-py3  
  
I also tried executing python directly, and `import` torch and tensorflow.  
However, the results of `torch.cuda.is_available()` and `tf.test.is_gpu_available` are still False.  
  
---  
Enter PT and execute `bash build.sh` to build image: lenovo-test  
If run it directly, the following information will appear in the end:  

![image](https://github.com/LYW0288/lenovo/blob/main/001.png)

