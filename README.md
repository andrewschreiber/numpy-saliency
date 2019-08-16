# numpy-saliency

Reference implementation of saliency map (aka attribution map)
techniques for deep learning in Python 3. Uses MNIST and LeNet 5. Tested with 3.6 & 3.7.


## Usage

Run

```
git clone https://github.com/andrewschreiber/numpy-saliency.git
cd numpy-saliency
# Activate your Python 3 environment
pip install -r requirements.txt
python main.py
```

You can train the model, test the model (using pretrained weights),
and/or generate a saliency map via uncommenting the code.

## Techniques

### Implemented

Vanilla Gradients


### Upcoming

Integrated Gradients

Adversarial Saliency maps

## Inspiration

https://github.com/utkuozbulak/pytorch-cnn-visualizations

https://github.com/gary30404
