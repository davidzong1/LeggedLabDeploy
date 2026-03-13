import torch

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
END = "\033[0m"  # 重置颜色


def policy_convert(
    old_policy_path: str,
    new_policy_path: str,
    keep_modules: list[str],
):
    """
    只保留 checkpoint 顶层的若干 key。
    keep_modules 例如：
        ["model_state_dict", "adapt_wrapper_dict"]
        或 ["model_state_dict"] 等。
    """
    ckpt = torch.load(old_policy_path, map_location="cpu")
    new_ckpt = {}

    # 1) 保留你想要的几个模块 dict
    for k in keep_modules:
        if k in ckpt:
            new_ckpt[k] = ckpt[k]
        else:
            print(f"{YELLOW}Key {k} not found in the checkpoint.{END}")

    # 3) 不保留 optimizer / rnd 等
    torch.save(new_ckpt, new_policy_path)
