import os
from os.path import join
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib as plt
from fast_slic import Slic
import kornia

from stego.utils import *
from stego.stego import STEGO
from stego.data import ContrastiveSegDataset

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="cfg", config_name="eval_wvn_config.yaml")
def my_app(cfg: DictConfig) -> None:
    result_dir = os.path.join(cfg.output_root, cfg.experiment_name)

    if cfg.save_vis:
        os.makedirs(join(result_dir, "img"), exist_ok=True)
        os.makedirs(join(result_dir, "label"), exist_ok=True)

    models = []
    for model_path in cfg.model_paths:
        models.append(STEGO.load_from_checkpoint(model_path))
    
    slic_models = []
    for n_clusters in cfg.slic_n_clusters:
        slic_models.append(Slic(num_components=n_clusters, compactness=cfg.slic_compactness))
    
    if cfg.save_vis:
        for n_clusters in cfg.stego_n_clusters:
            os.makedirs(join(result_dir, "stego_"+str(n_clusters)), exist_ok=True)
        for n_clusters in cfg.slic_n_clusters:
            os.makedirs(join(result_dir, "slic_"+str(n_clusters)), exist_ok=True)

    test_dataset = ContrastiveSegDataset(
        data_dir=cfg.data_dir,
        dataset_name=cfg.dataset_name,
        image_set="val",
        transform=get_transform(cfg.resolution, False, "center"),
        target_transform=get_transform(cfg.resolution, True, "center"),
        model_type="dino",
        resolution=cfg.resolution
    )

    test_loader = DataLoader(test_dataset, 1,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True, collate_fn=flexible_collate)

    for model in models:
        model.eval().cuda()

    model_metrics = [WVNMetrics("STEGO_"+str(i)+"/", i, 768, 70) for i in cfg.stego_n_clusters]
    slic_metrics = [WVNMetrics("SLIC_"+str(i)+"/", i, 768, 70) for i in cfg.slic_n_clusters]

    for i, batch in enumerate(tqdm(test_loader)):
        if cfg.n_imgs is not None and i >= cfg.n_imgs:
            break
        with torch.no_grad():
            img = batch["img"].squeeze()
            label = batch["label"].squeeze()

            if cfg.save_vis:
                image = Image.fromarray((kornia.utils.tensor_to_image(unnorm(img).cpu())*255).astype(np.uint8))
                image.save(join(result_dir, "img", str(i)+".png"))
                image = Image.fromarray((kornia.utils.tensor_to_image(label.cpu())*255).astype(np.uint8))
                image.save(join(result_dir, "label", str(i)+".png"))

            features = None
            code = None
            if len(models) > 0:
                features = models[0](batch["img"].cuda())[0]
                code = models[0].get_code(batch["img"].cuda())

            for model_index, model in enumerate(models):
                clusters, _ = model.postprocess(code=code, img=batch["img"], use_crf=cfg.run_crf)
                model_metrics[model_index].update(clusters.cuda(), label, features, code)
                if cfg.save_vis:
                    n_clusters = cfg.stego_n_clusters[model_index]
                    image = Image.fromarray((clusters.squeeze().numpy()*(255/n_clusters)).astype(np.uint8))
                    image.save(join(result_dir, "stego_"+str(n_clusters), str(i)+".png"))

            for model_index, model in enumerate(slic_models):
                img_np = kornia.utils.tensor_to_image(unnorm(img).cpu())
                clusters = model.iterate(np.uint8(np.ascontiguousarray(img_np) * 255))
                slic_metrics[model_index].update(torch.from_numpy(clusters), label.cpu(), features, code)
                if cfg.save_vis:
                    n_clusters = cfg.slic_n_clusters[model_index]
                    image = Image.fromarray((clusters*(255/n_clusters)).astype(np.uint8))
                    image.save(join(result_dir, "slic_"+str(n_clusters), str(i)+".png"))
                    
    for metric in model_metrics:
        tb_metrics = {
            **metric.compute(),
        }
        print(tb_metrics)

    for metric in slic_metrics:
        tb_metrics = {
            **metric.compute(),
        }
        print(tb_metrics)

    

if __name__ == "__main__":
    prep_args()
    my_app()