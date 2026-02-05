import os

from mineru.utils.config_reader import get_local_models_dir
from mineru.utils.enum_class import ModelPath


def auto_download_and_get_model_root_path(relative_path: str, repo_mode='pipeline') -> str:
    """
    支持文件或目录的可靠下载。
    - 如果输入文件: 返回本地文件绝对路径
    - 如果输入目录: 返回本地缓存下与 relative_path 同结构的相对路径字符串
    :param repo_mode: 指定仓库模式，'pipeline' 或 'vlm'
    :param relative_path: 文件或目录相对路径
    :return: 本地文件绝对路径或相对路径
    """

    local_models_config = get_local_models_dir()
    return local_models_config


if __name__ == '__main__':
    path1 = "models/README.md"
    root = auto_download_and_get_model_root_path(path1)
    print("本地文件绝对路径:", os.path.join(root, path1))
