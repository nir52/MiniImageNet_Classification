import os
import gc
import torch
import pytorch_lightning as pl
from datetime import datetime
import ViT_ as vit_py
from ViT_ import (
    DataModule, ViTClassifier, ClearCacheCallback, MinimalProgressBar,
    checkpoint_callback, early_stop_callback, logger, dataset_path,  augment_transform,
    DynamicAugmentationCallback, hf_image_processor, MODEL_NAME, BATCH_SIZE, NUM_WORKERS
)

def main():

    print("Mean:", vit_py.mean)
    print("Std:", vit_py.std)

    #Disabling symlinks warning while using huggingface transformers model
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print(f"[Main] PID={os.getpid()}")

    #Setting the seed before the dataset is initialized
    pl.seed_everything(42, workers=True)  # `workers=True` ensures worker-level reproducibility


    #Initialize the data module
    data_module = DataModule(
                        data_dir=dataset_path,
                        hf_image_processor=hf_image_processor,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        augment_transform=augment_transform
                    )

    data_module.setup() # Crucial to call setup here to get num_classes
    num_classes = data_module.num_classes
    class_names = data_module.class_names
    print(f"Number of classes: {num_classes}")

    # Initialize classifier
    model = ViTClassifier(
                        model_name=MODEL_NAME,
                        num_labels=num_classes,
                        class_labels=class_names, # Pass class_names here
                        learning_rate=5e-5   #Reduced from 1e-4
                    )

    #Calling garbage collection and empty CUDA cache before fitting the model
    gc.collect()
    torch.cuda.empty_cache()

    trainer = pl.Trainer(
                        logger=logger,
                        callbacks=[MinimalProgressBar(), checkpoint_callback, early_stop_callback, 
                                    ClearCacheCallback(), DynamicAugmentationCallback(data_module)],
                        default_root_dir="\\training_output",
                        max_epochs=10
    )

    # trainer = pl.Trainer(fast_dev_run=True)

    # trainer = pl.Trainer(
    #                     max_epochs=10,
    #                     overfit_batches=1,
    #                     logger=False,
    #                     enable_checkpointing=False,
    #                     accelerator="gpu",
    #                     devices=1,
    # )
    
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
