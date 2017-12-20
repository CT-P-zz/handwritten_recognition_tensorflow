# handwritten_recognition_tensorflow
OCR: handwritten recognition of words with tensorflow in python (CNN, LSTM)

# Project Title

Artificial Neural Network for handwritten words with Tensorflow in Python.

## Getting Started

Download the words dataset from http://www.fki.inf.unibe.ch/databases/iam-handwriting-database, extract to a 'words' folder inside 'Data' folder.


### Dependencies

Conda environment creation:

```

conda create –n (name_env)
conda install python=3.6
pip install --upgrade tensorflow-gpu
pip install Pillow

```

### Installing

A step by step:

```
python convert_images.py
python clean_IAM.py
python train.py
python test.py

```

