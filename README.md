# Video generative models using Interpretable Segmentation via Transformer Attention

## 🔧 Installation

Please follow the [FastVideo](https://github.com/hao-ai-lab/FastVideo/tree/main?tab=readme-ov-file#-installation) project for installation

## 🚀 Inference

```sh
# Download the model weight
python scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan-diffusers --local_dir=data/FastHunyuan-diffusers --repo_type=model
# CLI inference
bash scripts/inference/inference_hunyuan_hf_quantization.sh
```

## Acknowledgement

The project is built based on great [FastVideo](https://github.com/hao-ai-lab/FastVideo/tree/main), [Diffusers](https://github.com/huggingface/diffusers/tree/main), [ConceptionAttention](https://github.com/helblazer811/ConceptAttention/tree/master)