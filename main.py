import time
from cnn import *

def main():
    start_time = time.time() # Track training time

    train_in = "in-domain-train" # File paths assume image folders are in the same directory as the file
    train_out = "out-domain-train"
    val_in = "in-domain-eval"
    val_out = "out-domain-eval"

    model = learn(train_in, train_out)

    acc = compute_accuracy(val_in, model)
    print(f"ID Accuracy: {acc*100:.2f}%")

    acc2 = compute_accuracy(val_out, model)
    print(f"OOD Accuracy: {acc2*100:.2f}%")

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"Total training time: {minutes}m {seconds}s") 

if __name__ == "__main__":
    main()