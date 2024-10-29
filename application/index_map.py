from pathlib import Path
import hydra
from omegaconf import DictConfig
import sys
import os
sys.path.append(os.path.abspath('..'))
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    #data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    #data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    # set the root of the project

    data_dirs = "~/vlmaps"
    vlmap = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])
    #vlmap.load_map(data_dirs[config.scene_id])
    # vlmap.load_map(config.savepath, "vlmaps35.h5df")
    if config.map_config.model == "ovseg":
        vlmap.load_map("output/vlmap-ovseg", "vlmaps.h5df")
    elif config.map_config.model == "catseg-vitb":
        vlmap.load_map("output/vlmap-catseg-vitb", "vlmaps.h5df")
    elif config.map_config.model == "catseg-vitl":
        vlmap.load_map("output/vlmap-catseg-vitl", "vlmaps.h5df")
    elif config.map_config.model == "lseg":
        vlmap.load_map("output/vlmap-lseg", "vlmaps.h5df")
    elif config.map_config.model == "lseg-demo":
        vlmap.load_map("output/vlmap-lseg-demo", "vlmaps.h5df")
    elif config.map_config.model == "odise":
        vlmap.load_map("output/vlmap-odise", "vlmaps.h5df")
    else:
        raise ValueError("Invalid model name")
    visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb, voxel_size=.001)
    cat = input("What is your interested category in this scene?")
    # cat = "chair"

    
    if config.map_config.model in ['ovseg-vitb', 'catseg-vitb']:
        vlmap._init_clip(clip_version="ViT-B/16")
    elif config.map_config.model in ['ovseg-vitl', 'catseg-vitl']:
        vlmap._init_clip(clip_version="ViT-L/14")
    elif config.map_config.model == 'odise':
        vlmap._init_odise()
    else: # uses vitb-32 by default
        vlmap._init_clip()
    print("considering categories: ")
    print(mp3dcat[1:-1])
    if config.init_categories:
        vlmap.init_categories(mp3dcat[1:-1], model_name=config.map_config.model)
        mask = vlmap.index_map(cat, with_init_cat=False, model_name=config.map_config.model)
    else:
        mask = vlmap.index_map(cat, with_init_cat=False, model_name=config.map_config.model)

    if config.index_2d:
        mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, config.params.gs)
        rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)
        visualize_masked_map_2d(rgb_2d, mask_2d)
        heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
        visualize_heatmap_2d(rgb_2d, heatmap)
    else:
        visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb)
        heatmap = get_heatmap_from_mask_3d(
            vlmap.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
        )
        visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb)


if __name__ == "__main__":
    main()
