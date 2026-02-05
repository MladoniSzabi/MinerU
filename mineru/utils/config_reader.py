# Copyright (c) Opendatalab. All rights reserved.
import json
import os
from loguru import logger

try:
    import torch
    import torch_npu
except ImportError:
    pass


# 定义配置文件名常量
CONFIG_FILE_NAME = os.getenv('MINERU_TOOLS_CONFIG_JSON', 'mineru.json')
model_dirs = os.path.join(os.getcwd(), "models")


def read_config():
    global model_dirs
    return {
        "models-dir": model_dirs
    }


def get_device():
    device_mode = os.getenv('MINERU_DEVICE_MODE', None)
    if device_mode is not None:
        return device_mode
    else:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            try:
                if torch_npu.npu.is_available():
                    return "npu"
            except Exception as e:
                pass
        return "cpu"


def get_formula_enable(formula_enable):
    formula_enable_env = os.getenv('MINERU_FORMULA_ENABLE')
    formula_enable = formula_enable if formula_enable_env is None else formula_enable_env.lower() == 'true'
    return formula_enable


def get_table_enable(table_enable):
    table_enable_env = os.getenv('MINERU_TABLE_ENABLE')
    table_enable = table_enable if table_enable_env is None else table_enable_env.lower() == 'true'
    return table_enable


def get_latex_delimiter_config():
    config = read_config()
    if config is None:
        return None
    latex_delimiter_config = config.get('latex-delimiter-config', None)
    if latex_delimiter_config is None:
        # logger.warning(f"'latex-delimiter-config' not found in {CONFIG_FILE_NAME}, use 'None' as default")
        return None
    else:
        return latex_delimiter_config


def get_llm_aided_config():
    config = read_config()
    if config is None:
        return None
    llm_aided_config = config.get('llm-aided-config', None)
    if llm_aided_config is None:
        # logger.warning(f"'llm-aided-config' not found in {CONFIG_FILE_NAME}, use 'None' as default")
        return None
    else:
        return llm_aided_config


def get_local_models_dir():
    config = read_config()
    if config is None:
        return None
    models_dir = config.get('models-dir')
    if models_dir is None:
        logger.warning(
            f"'models-dir' not found in {CONFIG_FILE_NAME}, use None as default")
    return models_dir
