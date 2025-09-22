
## Preparation

### Environment Setup

```bash
git clone https://github.com/mit-han-lab/vila-u
cd vila-u
./environment_setup.sh vila-u
```

### Download Models

Please download our [models](https://huggingface.co/collections/mit-han-lab/vila-u-7b-6716f7dd5331e4bdf944ffa6) from HuggingFace.

```bash
git lfs install
git clone https://huggingface.co/mit-han-lab/vila-u-7b-256
```

## Usage

### Gradio Demo

Run the following command to launch a local gradio demo:
```bash
CUDA_VISIBLE_DEVICES=0 python app.py --model_path path/to/your_downloaded_model
```

### Command Line Inference

```bash
# Image Understanding
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path path/to/your_downloaded_model --image_path assets/example_image1.jpg --query "Can you describe what is happening?"
```

```bash
# Video Understanding
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path path/to/your_downloaded_model --video_path assets/example_video1.mp4 --query "Elaborate on the visual and narrative elements of the video in detail."
```

```bash
# Image Generation
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path path/to/your_downloaded_model --prompt "A snowy mountain." --save_path path/to/save_images --generation_nums 8
```

```bash
# Video Generation
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path path/to/your_downloaded_model --prompt "Fireworks in the air." --video_generation True --save_path path/to/save_videos
```

### Evaluation

Evaluate VILA-U on visual language benchmarks with the following command:
```bash
vila_u-eval -m path/to/model -c vicuna_v1 -ti local
```
Please refer to `vila_u/cli/eval.py` for more argument details.

### Training

Note: Please prepare data before training. Data preparation details are in the file `vila_u/data/datasets_mixture.py`.


```bash
# SFT
CUDA_VISIBLE_DEVICES=0 python -m vila_u.train.train \
  --model_name_or_path /home/choi/ToT_vla/vila-u/vila-u-7b-256 \
  --output_dir /home/choi/ToT_vla/vila-u/out/covla-sft-test2 \
  --data_mixture covla_dataset_mini \
  --num_video_frames 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --mm_projector_lr 1e-4 \
  --tune_language_model True \
  --tune_vision_tower False \
  --tune_mm_projector True \
  --optim adamw_bnb_8bit \
  --bf16 True \
  --gradient_checkpointing True \
  --num_train_epochs 1 \
  --max_steps 100 \
  --save_steps 100 \
  --logging_steps 1 \
  --overwrite_output_dir True \
  --save_safetensors False
```


