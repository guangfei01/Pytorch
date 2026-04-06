# Train a PyTorch model using all model files created

import os
import torch
from torch import nn
from torch import optim
import data_setup, get_data, model_builder, engine, utils
from torchvision import transforms

NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_UNITS = 10

def argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="Directory containing training data")
    parser.add_argument("--test_dir", type=str, help="Directory containing test data")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, help="Batch size for dataloaders")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer")
    parser.add_argument("--hidden_units", type=int, help="Number of hidden units in model")
    args = parser.parse_args()
    return args

def main():
    args = argparse()
    train_dir = args.train_dir
    test_dir = args.test_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units


       # setup directories
    #    train_dir = "/Users/jjgwerty/Documents/gj/learning/pytorch/zero2master_pytorch/going_modular/data/pizza_steak_sushi/train"
    #    test_dir = "/Users/jjgwerty/Documents/gj/learning/pytorch/zero2master_pytorch/going_modular/data/pizza_steak_sushi/test"

       # setup target device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    data_transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                                    test_dir,
                                                                                    transform=data_transform,
                                                                                    batch_size=batch_size)

    model = model_builder.TinyVGG(input_shape=3,
                                    hidden_units=hidden_units,
                                    output_shape=len(class_names)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=num_epochs,
                device=device)

    utils.save_model(model=model,
                    target_dir="../models",
                    model_name="05_going_modular_script_mode_tinyvgg_model.pth")


if __name__ == "__main__":
    # args = argparse()
    main()