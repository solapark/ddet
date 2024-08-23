from mmcv.utils import Registry, build_from_cfg

BBOX_DLT = Registry('bbox_dlt')

def build_dlt(cfg, **default_args):
    """Builder of box assigner."""
    return build_from_cfg(cfg, BBOX_DLT, default_args)
