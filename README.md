# audio_adversarial_training

Milestone: In this itera- tion of our project, we demonstrate the feasibil- ity of adversarial robustness in this domain by training adversarially using a simple speech-to- text dataset.

## Install:

This project requires Python and the following Python libraries installed:

* numpy
* PyTorch
* Scipy

## Code:

The models are run in the audio_training.ipynb notebook. For loading the data use dataloader.py and CustomDataset.py to generate the PyTorch dataloader. The attack functions are stored in attacks.py, and the basetrainer class for training the model is found in basetrainer.py. The ResNet34 architecture is found in the "models" folder, and examples of audio reconstruction from spectrograms is found in Audio Reconstruction.ipynb

## Trained Models:

Trained models are found in the "saved" folder which can be loaded using

```
torch.load(model_name)
```

