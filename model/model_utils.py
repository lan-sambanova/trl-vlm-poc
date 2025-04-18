def get_image_seqlen(config):
    return (
        (config.vision_config.image_size // config.vision_config.patch_size) ** 2 + 1
    ) * config.vision_config.max_num_tiles
