import torch

from collections import OrderedDict
from typing import Any, Dict, List, Tuple


def get_state_dict(checkpoint: Dict[str, Any]) -> OrderedDict[str, Any]:
    if "state_dict" in checkpoint:
        dd = checkpoint["state_dict"]
    else:
        dd = checkpoint["model"]
    assert isinstance(dd, OrderedDict)
    return dd


def right_justify_integer(value: int, max_value: int) -> str:
    """
    Right-justifies an integer, padding with spaces so that its width
    matches the maximum width determined by the maximum value.

    Args:
        value (int): The integer to format.
        max_value (int): The maximum possible value, which determines the width.

    Returns:
        str: The right-justified, space-padded string representation of the integer.
    """
    # Calculate the width based on the number of digits in max_value
    max_width = len(str(max_value))

    # Format the value as a right-justified string with spaces
    return f"{value:>{max_width}}"


def print_keys(dd: OrderedDict[str, Any]) -> None:
    assert isinstance(dd, OrderedDict)
    counter: int = 1
    max_int: int = len(dd) + 1
    max_key_length = 0
    for k in dd.keys():
        max_key_length = max(max_key_length, len(k))
    max_key_length += 4
    for k, p in dd.items():
        print(
            f"{right_justify_integer(counter, max_int)}: {k:<{max_key_length}}  {p.shape}"
        )
        counter += 1
    print(f"^ {len(dd)} items")


def drop_prefixed_params(
    dd: OrderedDict[str, Any], prefix: str
) -> OrderedDict[str, Any]:
    if not prefix:
        return dd
    results: OrderedDict[str, Any] = OrderedDict()
    for k, v in dd.items():
        if not k.startswith(prefix):
            results[k] = v
    return results


def replace_param_name_portion(
    dd: OrderedDict[str, Any], from_str: str, to_str: str, beginning_only: bool
) -> OrderedDict[str, Any]:
    if from_str == to_str:
        return dd
    results: OrderedDict[str, Any] = OrderedDict()
    for k, v in dd.items():
        if beginning_only:
            if k.startswith(from_str):
                k = to_str + k[len(from_str) :]
        elif from_str in k:
            k = k.replace(from_str, to_str)
        assert k not in results
        results[k] = v
    return results


REPLACE_MAP: List[Tuple[str, str, bool]] = [
    (".stage1.", ".dark2.", False),
    (".stage2.", ".dark3.", False),
    (".stage3.", ".dark4.", False),
    (".stage4.", ".dark5.", False),
    (".main_conv.", ".conv1.", False),
    (".short_conv.", ".conv2.", False),
    (".final_conv.", ".conv3.", False),
    (".blocks.", ".m.", False),
    ("neck.reduce_layers.0.", "backbone.lateral_conv0.", True),
    ("bbox_head.multi_level_cls_convs.", "head.cls_convs.", True),
    ("bbox_head.multi_level_reg_convs.", "head.reg_convs.", True),
    ("bbox_head.multi_level_conv_cls.", "head.cls_preds.", True),
    ("bbox_head.multi_level_conv_reg.", "head.reg_preds.", True),
    ("bbox_head.multi_level_conv_obj.", "head.obj_preds.", True),
    ("neck.out_convs.", "head.stems.", True),
    ("neck.reduce_layers.1.", "backbone.reduce_conv1.", True),
    ("neck.top_down_blocks.0.", "backbone.C3_p4.", True),
    ("neck.top_down_blocks.1.", "backbone.C3_p3.", True),
    ("neck.bottom_up_blocks.0.", "backbone.C3_n3.", True),
    ("neck.bottom_up_blocks.1.", "backbone.C3_n4.", True),
    ("neck.downsamples.0.", "backbone.bu_conv2.", True),
    ("neck.downsamples.1.", "backbone.bu_conv1.", True),
]


def get_exclusion_keys(
    keys_1: List[str], keys_2: List[str]
) -> Tuple[List[str], List[str]]:
    ex_1: List[str] = []
    ex_2: List[str] = []
    es1 = set(keys_1)
    assert len(es1) == len(keys_1)
    es2 = set(keys_2)
    assert len(es2) == len(keys_2)
    for k1 in keys_1:
        if k1 not in es2:
            ex_1.append(k1)
    for k2 in keys_2:
        if k2 not in es1:
            ex_2.append(k2)
    return ex_1, ex_2


def print_keys_and_shapes(keys: List[str], state_dict: OrderedDict[str, Any]) -> None:
    dd: OrderedDict[str, Any] = OrderedDict()
    for k in keys:
        dd[k] = state_dict[k]
    if dd:
        print_keys(dd)


def fix_params(input: str, pattern: str, output: str):
    input_checkpoint = torch.load(input)
    input_state_dict = get_state_dict(input_checkpoint)
    input_state_dict = drop_prefixed_params(input_state_dict, "ema_")
    input_state_dict = replace_param_name_portion(
        input_state_dict, "backbone.", "backbone.backbone.", beginning_only=True
    )
    for from_str, to_str, beginning_only in REPLACE_MAP:
        input_state_dict = replace_param_name_portion(
            input_state_dict, from_str, to_str, beginning_only=beginning_only
        )

    pattern_checkpoint = torch.load(pattern)
    pattern_state_dict = get_state_dict(pattern_checkpoint)

    # TEMP: focus on backbone only
    # input_state_dict = drop_prefixed_params(input_state_dict, "neck.")
    # input_state_dict = drop_prefixed_params(input_state_dict, "bbox_head.")
    # pattern_state_dict = drop_prefixed_params(pattern_state_dict, "head.")

    # input_state_dict = drop_prefixed_params(input_state_dict, "backbone.")
    # pattern_state_dict = drop_prefixed_params(input_state_dict, "backbone.")

    # print_keys(input_state_dict)
    # print_keys(pattern_state_dict)

    kk1 = list(input_state_dict.keys())
    kk2 = list(pattern_state_dict.keys())
    assert len(kk1) == len(kk2)

    k1_ex, k2_ex = get_exclusion_keys(kk1, kk2)
    assert len(k1_ex) == len(k2_ex)

    print_keys_and_shapes(k1_ex, input_state_dict)
    print_keys_and_shapes(k2_ex, pattern_state_dict)

    # If this assertion fails, you probably caused a duplicate key with the substitution
    min_l = min(len(input_state_dict), len(pattern_state_dict))
    # assert len(input_state_dict) == len(pattern_state_dict)
    mismatch_count: int = 0
    for index, (k1, k2) in enumerate(zip(kk1[:min_l], kk2[:min_l])):
        if k1 == k2:
            assert pattern_state_dict[k1].shape == input_state_dict[k2].shape
        elif k1 in kk2:
            assert pattern_state_dict[k1].shape == input_state_dict[k1].shape
        else:
            if not mismatch_count:
                print(f"\n\nFirst mismatch at param # {index + 1}: {k1} != {k2}")
            mismatch_count += 1
    if mismatch_count:
        print(f"\nMismatch count: {mismatch_count}")

    final_pattern_checkpoint: Dict[str, Any] = torch.load(pattern)
    assert len(final_pattern_checkpoint["model"]) == len(input_state_dict)
    assert set(final_pattern_checkpoint["model"].keys()) == set(input_state_dict.keys())
    del final_pattern_checkpoint["model"]
    final_pattern_checkpoint["model"] = input_state_dict
    torch.save(final_pattern_checkpoint, output)


if __name__ == "__main__":
    input_file: str = "yolox_s_8x8_300e_coco_80e_ch.pth"
    pattern_file: str = "yolox_s.pth"
    output_file: str = "test_output_yolox_s_coco300_ch80.pth"
    fix_params(input_file, pattern_file, output_file)
    print("Done.")
