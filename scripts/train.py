from torch.utils.data import DataLoader
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from stego.stego import STEGO
from stego.utils import *
from stego.data import ContrastiveSegDataset



@hydra.main(config_path="cfg", config_name="train_config.yaml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    seed_everything(seed=0)

    # model = STEGO(cfg.num_classes).cuda()
    model = STEGO.load_from_checkpoint(cfg.model_path).cuda()
    model.cfg.lr = cfg.lr
    model.cfg.val_n_imgs = cfg.val_n_imgs

    train_dataset = ContrastiveSegDataset(
        data_dir=cfg.data_dir,
        dataset_name=cfg.dataset_name,
        image_set="train",
        transform=get_transform(cfg.resolution, False, "center"),
        target_transform=get_transform(cfg.resolution, True, "center"),
        model_type=model.backbone_name,
        resolution=cfg.resolution,
        num_neighbors=cfg.num_neighbors,
        pos_images=True,
        pos_labels=True
    )

    val_dataset = ContrastiveSegDataset(
        data_dir=cfg.data_dir,
        dataset_name=cfg.dataset_name,
        image_set="val",
        transform=get_transform(cfg.resolution, False, "center"),
        target_transform=get_transform(cfg.resolution, True, "center"),
        model_type=model.backbone_name,
        resolution=cfg.resolution
    )

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    wandb_logger = WandbLogger(project=cfg.wandb_project, name=cfg.wandb_name, log_model=cfg.wandb_log_model)

    trainer = Trainer(
        logger=wandb_logger,
        max_steps=cfg.max_steps,
        default_root_dir=cfg.checkpoint_dir,
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                every_n_train_steps=1,
                save_top_k=2,
                monitor="val/cluster/mIoU",
                mode="max",
            )
        ],
        gpus=1
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()