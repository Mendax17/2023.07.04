import tensorflow as tf
print(tf.__version__) # 2.7.4

gpus=tf.config.experimental.list_physical_devices('GPU')
print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')] # 0번째 GPU라는 의미이다. # CUDA로 잡히는건 NVIDIA GPU만 잡힌다.

if gpus:
    print("GPU ON")
else:
    print("GPU OFF")