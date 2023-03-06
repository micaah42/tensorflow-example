# Tensorflow Lite Example

This repo contains a simple, but somewhat useful example for an embedded linux machine learning algorithm.
It should illustrate some concepts suitable for working with neural networks on image processing

The goal is to detect the rotation of scanned or photographed documents and 
use it to rotate them back.

As a dataset we will use a pdf scriptum which we crop and rotate for training and testing samples and labels.

## Installation
* clone repo: `git clone https://github.com/micaah42/tensorflow-example.git`
* before you install the other dependencies, 
* install dependencies: `pip install -r requirements.txt`
* prepare resources folder: create a folder `resources` in the repo (you can use symlinks to save data to other disks)

## Usage

### Training/Development 
* start jupyter server: `jupyter notebook`
* navigate to `notebooks/cnn.ipynb` and run all cells

### Inference
* run 

