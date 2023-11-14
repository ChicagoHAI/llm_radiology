# Training Vision Model Instructions

## Configuration File Setup
Ensure the correct configuration file path is set by modifying the `VitChexpert.cfg` located at:

```python
cfg_path = './config/VitChexpert.cfg'
```

Remember to update any relevant paths in the configuration file to correspond with your project structure.

## Weights & Biases (wandb) Configuration
Modify `train.py` to configure wandb settings:

- To use wandb, confirm that your settings are accurate.
- If wandb is not needed, comment out its settings to bypass this feature.

## Running the Training Process
To begin training, run the following command in your terminal:

```bash
python train.py
```

# Image Model Evaluation

For evaluating your image model, specify the checkpoint path in `evaluation.py`. Execute the evaluation using:

```bash
python evaluation.py
```

# Retrieval System Integration

Update the vision model path in the integration command, which can be found in the [main repo README](https://github.com/ChicagoHAI/llm_radiology/tree/main#generating-reports-with-pragmatic-llama).

# Vision Model Trained Checkpoint

We provide the checkpoint of our trained vision model 

[Vision Model Checkpoint](https://drive.google.com/file/d/12l-SeZ-SSR8xtVCVlJ0wDHwxVZ3iNcmh/view?usp=drive_link)
