import os
import time
import torch
import models
import utils
import pandas as pd
# from tqdm.notebook import tqdm

__name2metric__ = {"multi_label": "MultiLabelAccuracyMetrics", "multi_class":"MultiClassAccuracyMetrics"}

backbones_dict = {"x3d": ("Recognizer3D", ["mlh", "mcsh", "mcmh", "mcmha"], False, "CTHW"), 
                  "mvit": ("Recognizer3D", ["mlh", "mcsh"], True, "CTHW"), 
                  "i3d": ("Recognizer3D", ["mlh", "mcsh", "mcmh", "mcmha"], False, "CTHW"), 
                  "swin": ("Recognizer3D", ["mlh", "mcsh", "mcmh", "mcmha"], False, "CTHW"),
                  "tsm": ("Recognizer2D", ["mlh", "mcsh", "mcmh", "mcmha"], False, "TCHW")}

#num_labels/num_class, classification head type
head_dict = {"mlh": (7, "multi_label"),
              "mcsh": (9, "multi_class"), 
              "mcmh": (3, "multi_class"), 
              "mcmha": (3, "multi_class")}


@torch.no_grad()
def test_results(data_type, fmt, recognizer, backbone, head, num_labels, is_2d):
    val_loader = utils.create_video_dataloader(root_path="/mnt/", 
                                               data_list=data_type+'_test.txt', 
                                               batch_size=32,
                                               is_train=False, 
                                               format=fmt,
                                               num_worker=16)
    
    model = getattr(models, recognizer)(backbone = backbone, 
                                        cls_head = head,
                                        num_labels = num_labels,
                                        dropout_rate=0,
                                        is_3d = not is_2d)
    model.cuda();
    model.eval();
    state_dict = torch.load(os.path.join("runs/default_aug/", f"{backbone}_{head}_SGD", 'ckpt', "best_val.pth"))["state_dict"]
    model.load_state_dict(state_dict)

    metrics_h = getattr(utils, __name2metric__[data_type])(criteria="hamming", threshold=0.5)
    metrics_e = getattr(utils, __name2metric__[data_type])(criteria="exact_match", threshold=0.5)

    for idx, batch in enumerate(val_loader):
        inp = batch[0].cuda()
        trg = batch[1].cuda()
        
        out = model(inp)
        _ = metrics_h(out, trg)
        _ = metrics_e(out, trg)
    
    del model, val_loader, batch, state_dict
    return metrics_h.compute().item(), metrics_e.compute().item()


def main():
    indexes = ["tsm", "i3d", "x3d", "mvit", "swin"]
    columns = ["mlh", "mcsh", "mcmh", "mcmha"]

    df_exact_match = pd.DataFrame(index=indexes, columns=columns)
    df_hamming_match = pd.DataFrame(index=indexes, columns=columns)

    for backbone, v in backbones_dict.items():
        recognizer, heads, is_2d, fmt = v
        for head in heads:
            print(f"Starting {backbone}_{head}")
            tsrt = time.perf_counter()
            num_labels, data_type = head_dict[head]
            
            df_hamming_match.loc[backbone, head], df_exact_match.loc[backbone, head] = test_results(data_type, fmt, recognizer, backbone, head, num_labels, is_2d)
            torch.cuda.empty_cache()
            print(f"Finished {backbone}_{head}: {time.perf_counter() - tsrt}")

    df_hamming_match.to_pickle("defaultaug_hamming.pkl")
    df_exact_match.to_pickle("defaultaug_extmch.pkl")


if __name__ == '__main__':
    main()

    