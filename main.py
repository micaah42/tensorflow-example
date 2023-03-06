import io
import logging
import os
import sys
import argparse

import numpy as np
from PIL import Image

if os.uname().machine == 'x86_64':
    import tensorflow.lite as interpreter_module
else:
    import tflite_runtime.interpreter as interpreter_module


class TFLiteModel:
    def __init__(self, filename, threads=2):
        self.interpreter = interpreter_module.Interpreter(filename, num_threads=threads)
        self.interpreter.allocate_tensors()

        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()

    def __call__(self, x: np.ndarray):
        self.interpreter.set_tensor(self.inputs[0]['index'], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.outputs[0]['index'])


def read_from_stdin():
    while True:
        buffer = sys.stdin.buffer.read()
        if buffer:
            yield buffer


def read_files(files):
    for file in files:
        with open(file, 'rb') as f:
            yield f.read()


def split_image(image: Image, input_size: int):
    image = np.asarray(image)
    images = np.asarray(np.array_split(image[:-(image.shape[0] % input_size)], image.shape[0] // input_size, axis=0))
    images = np.asarray([np.array_split(image[:, :-(image.shape[1] % input_size)], image.shape[1] // input_size, axis=1)
                         for image in images])
    return images.reshape((np.prod(images.shape[:2]), *images.shape[2:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', type=str, help='file to get rotation of. if none read from std in')
    parser.add_argument('--threads', type=int, default=2, help='number of threads for inference')
    parser.add_argument('--model', '-m', type=str, help='path to model file. default is latest')
    args = parser.parse_args()

    if args.model is None:
        models_path = f'{"/".join(__file__.split("/")[:-1])}/models'
        models = os.listdir(models_path)
        args.model = f'{models_path}/{max(models)}/model.tflite'
        logging.warning(f'model: {args.model}')

    inputs = read_files(args.files) if args.files else read_from_stdin()
    model = TFLiteModel(args.model, args.threads)

    for buffer in inputs:
        # preprocess bytes input into image
        image = Image.open(io.BytesIO(buffer))
        image = image.convert('F')
        array = np.asarray(image, dtype=np.float32) / 255.

        # extract samples of the proper input size
        samples = split_image(array, model.inputs[0]['shape'][1])
        samples = samples[np.argsort(samples.mean(axis=-1).mean(axis=-1))[:5]]
        samples = np.expand_dims(samples, axis=-1)

        # select a couple of sections of the right input size
        prediction = np.array([model([sample]) for sample in samples])
        prediction = 360 * prediction.sum(axis=0).argmax() / model.outputs[0]['shape'][1]
        image = image.rotate(-prediction, expand=True)
        image.show()
        print(prediction)


if __name__ == '__main__':
    main()
