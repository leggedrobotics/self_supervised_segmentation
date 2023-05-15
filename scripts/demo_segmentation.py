import hydra
import torch.multiprocessing
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from stego.stego import *


torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(config_path="cfg", config_name="demo_config.yaml")
def my_app(cfg: DictConfig) -> None:
    result_dir = os.path.join(cfg.output_root, cfg.experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "linear"), exist_ok=True)

    model = STEGO.load_from_checkpoint(cfg.model_path)
    print(OmegaConf.to_yaml(model.cfg))

    dataset = UnlabeledImageFolder(
        root=cfg.image_dir,
        transform=get_transform(cfg.resolution, False, "center"),
    )

    loader = DataLoader(dataset, cfg.batch_size * 2,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()
    cmap = create_cityscapes_colormap()

    for i, (img, name) in enumerate(tqdm(loader)):
        with torch.no_grad():
            img = img.cuda()
            cluster_crf, linear_crf = model(img)
            for j in range(img.shape[0]):
                new_name = ".".join(name[j].split(".")[:-1]) + ".png"
                Image.fromarray(cmap[linear_crf[j]].astype(np.uint8)).save(os.path.join(result_dir, "linear", new_name))
                Image.fromarray(cmap[cluster_crf[j]].astype(np.uint8)).save(os.path.join(result_dir, "cluster", new_name))



if __name__ == "__main__":
    prep_args()
    my_app()