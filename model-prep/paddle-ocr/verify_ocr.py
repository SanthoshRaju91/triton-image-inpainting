import paddle

print(f"Paddle version: {paddle.__version__}")

gpu_available = paddle.is_compiled_with_cuda()

if gpu_available:
    try:
        paddle.device.set_device('gpu')
        print("GPU device detected and available.")

        place = paddle.CUDAPlace(0)
        print(f"Device name: {paddle.device.cuda.get_device_name(place)}")
    except Exception as e:
        print(f"Error initialising GPU: {e}")
        print("Please check CUDA toolkit and cuDNN installation and compatibility")
else:
    print("PaddlePaddle was not compiled with CUDA support or CUDA is not found")