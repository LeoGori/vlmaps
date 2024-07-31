from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap

import torch
import gc

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig) -> None:

    # free up GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    

    vlmap = VLMap(config.map_config)
    #data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    #data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    vlmap.create_map_ros()


if __name__ == "__main__":
    main()
