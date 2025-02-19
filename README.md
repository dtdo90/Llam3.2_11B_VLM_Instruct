# Llam3.2_11B_VLM_Instruct

**Fine tune LLAMA 3.2-11B Vision Language Model for  Multi-Turn Image Conversations**

This repository demonstrates the fine-tuning of **LLAMA 3.2-11B VLM** to improve its ability to engage in multi-turn image-based conversations. The model is trained to understand images and answer follow-up questions about them. The dataset is sourced from HuggingFaceM4/the_cauldron.


## Training Details
- **Framework**: SFTTrainer from the trl (Transformer Reinforcement Learning) library.
- **Parameter Optimization**: QLoRA (Low-Rank Adaptation) to reduce memory usage while maintaining model performance.
- **Training Setup**:
    - Duration: ~3 hours for 1 epoch (2,000 samples)
    - Hardware: NVIDIA A6000 GPU (48GB VRAM)
