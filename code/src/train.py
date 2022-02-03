import os
import shutil
import argparse
import mlflow.fastai
import fastai.vision as vis

def parse_args():
    parser = argparse.ArgumentParser(description="Fasti.ai MNIST example")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5). Note it takes about 1 min per epoch")
    parser.add_argument("--data_path", type=str, default='data/', help="Directory path to training data")
    parser.add_argument("--model_path", type=str, default='outputs/', help="Model output directory")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    print(args.data_path)
    os.listdir(os.getcwd())

    # Make sure model output path exists
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Extract training data
    data_file = os.path.join(args.data_path, 'mnist_tiny.tgz')
    shutil.unpack_archive(filename=data_file, format="gztar")
    
    # Prepare, transform, and normalize the data
    data = vis.ImageDataBunch.from_folder(
        "./mnist_tiny/", ds_tfms=(vis.rand_pad(2, 28), []), bs=64
    )
    data.normalize(vis.imagenet_stats)

    # Train and fit the Learner model, set output path to args.model_path
    learner = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy, path=args.model_path)

    # Train and fit with default or supplied command line arguments
    learner.fit(args.epochs, 0.01)
    
    # Export model
    learner.export(file='model.pkl')

if __name__ == "__main__":
    main()