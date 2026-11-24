"""
Fine-tune a pretrianed model that already imported from timm model on the specule

WHAT THIS FILE DOES:
   1. Read the manifest.csv file one row at a time
   2. Splits the image into training and validation group
   3. Loads a pretrain model and fine-tunes if on your species.
   4. Print accuracy after every set of epochs runs and save the best model to the models/ 

For a quick check on pipeline: python src/train.py --sanity --epochs 15

Run, Note: Its needs a speciment-id column in data/manifest.csv):
    python src/train.py --model mobilenetv3_large_100
    python src/train.py --model efficientnet_b0
"""
import random
import argparse
import time 
from pathlib import Path

import timm 
import torch 
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader

from dataset import SpiculeDataset          # loads, pads and augment images 

def run_epoch(model, loader, loss_fn, device, optimizer=None):
   """Go through the data 

   training, validation and returns the average loss and the accuracy.
   """
   
   training = optimizer is not None 
   model.train(training)
   total_loss, correct, n = 0.0, 0, 0

   # Gradients only needed for training
   with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item() # the highest scorre is the prediction 
            n += len(y)

   return total_loss / max(n, 1), correct / max(n, 1)

def main():
    #-----------------------------------------------------------------
    # 1. Enable settings that could be changed from your command line. 
    #-----------------------------------------------------------------
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--model", default="mobilenetv3_large_100",
                   help="any timm model name, e.g efficientnet_b0")
    p.add_argument("--sclere-type", default=None,
                   help="only use one sclere type, e.g megasclere")
    p.add_argument("--epochs", type=int, default=10,
                   help="how many times the model sees the whole training set")
    p.add_argument("--batch-size", type=int, default=8,
                   help="the number of images processed at once, we keep it small for GPU")
    p.add_argument("--size", type=float, default=224,
                   help="Resize all images to 224 and you could later do 384 or 512")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="learning rate and you could alway change it depending on results")
    p.add_argument("--seed", type=int, default=42,
                   help="fixed random seed so results can be repeated")
    p.add_argument("--sanity", action="store_true",
                   help="overfit a few images to test the code")
    args = p.parse_args()

    #fix the random seed 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #---------------------------------------------
    # 2. Read list of species from manifest file.
    #---------------------------------------------
    df = pd.read_csv(args.manifest)
    if args.sclere_type:
        df = df[df["sclere_type"] == args.sclere_type]

    # Model output numbers
    classes = sorted(df["species"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"{len(df)} images, {len(classes)} species")

    #--------------------------------------------------
    # 3. Split images into training and validation. 
    #--------------------------------------------------
    if args.sanity:
        # Code check only: If the conde is right, the accuracy should climb to a 100%
        train_df = df.groupby("species").head(2)
        val_df = train_df
        print(f"SANITY MODE: {len(train_df)} images. Training accuracy should approach 100%.")
    else:
        if "speciment_id" not in df.columns:
            raise SystemExit(
                "The manifest has no 'speciment-id' column. Splitting by image would leak "
                "specimens between train and validation. Add specimen IDs first, "
                "or run with --sanity."
            )
        # Split by specimen, to avoid the model from recognising what it already saw.
        # TODO: later replace this single split with cross-validation 
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(df, groups=df["speciment_id"]))
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        print(f"train: {len(train_df)} images, validation: {len(val_df)} images")

    #------------------------------------------------------------
    # 4. Data loader ( a batch of images handed to the model )
    #------------------------------------------------------------
    # Note that if the train=True it will turn on the augmentation 
    # Also check for sanity
    train_ds = SpiculeDataset(train_df, class_to_idx, args.size, train=not args.sanity)
    val_ds = SpiculeDataset(val_df, class_to_idx, args.size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,                              # shuffle new images for every image
        drop_last=len(train_ds) > args.batch_size,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    #------------------------------------------------------
    # 5. Build our model
    #------------------------------------------------------
    # Uses the GPU automatically if one exist; otherwise fall back to your cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Pretrained=True then download the weight of the models installed 
    # num_classes replaces the final layer so it outputs one score per species.
    model = timm.create_model(args.model, 
                              pretrained=True, 
                              num_classes=len(classes)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    # TODO: if some species have far fewer images than other, add class weights.

    #------------------------------------------------------------------------------------
    # 6. Train, check, and save the best model ## Experiment, experiment, experiment......
    #-------------------------------------------------------------------------------------
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, device, optimizer) # learn
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, device)
        print(
            f"epoch {epoch:02d}, train loss {tr_loss:.3f} acc {tr_acc:.2f} "
            f"val loss {va_loss:.3f} acc {va_acc:.2f} ({time.time() - start:.0f}s)"
        )

        # Keep the version of the model that scored best on validation.
        # (Skipped in sanity mode, since that run is only a code check.)
        if not args.sanity and va_acc >= best_acc:
            best_acc = va_acc
            Path("models").mkdir(exist_ok=True)
            # Save the class name with the weights so the API knows which number is which species.
            torch.save({"state_dict": model.state_dict(), "classes": classes},
                       f"models/{args.model}.pt")
    if not args.sanity:
        print(f"best validation accuracy: {best_acc:.2f} (saved to models/{args.model}.pt)")

if __name__ == "__main__":
   main()