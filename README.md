# triton-image-inpainting

sudo docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002  -v /home/santhoshraju/workspace/image-inpainting/model-repository:/models  nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models --log-verbose=1


python ocr_client_multilingual.py --lang en --image quiz.jpg

python triton_pipeline_client.py --lang en --image sample.png

python bin/predict.py model.path=$(pwd)/LaMa_models/big-lama indir=$(pwd)/small_input_images outdir=$(pwd)/output
 1271  python bin/predict.py model.path=$(pwd)/LaMa_models/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output