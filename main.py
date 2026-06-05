import random

import torch
from torch.utils.data import DataLoader

from config import Config
from data import (NeuDataset, build_label_map, build_splits,
                  session_samples, test_samples, get_transforms)
from engine import compute_metrics, evaluate, imprint_new_class, train_base
from model import FSCILModel


def main():
    cfg = Config()
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    label_map = build_label_map(cfg)
    train_pool, test = build_splits(cfg)
    base_samples, inc_sessions = session_samples(train_pool, cfg, rng)
    train_t, eval_t = get_transforms(cfg)

    # ---- Session 0: base training ----
    base_ds = NeuDataset(base_samples, label_map, train_t)
    base_loader = DataLoader(base_ds, batch_size=cfg.batch, shuffle=True,
                             num_workers=cfg.workers, drop_last=True)

    seen = list(cfg.base_classes)
    base_test_loader = DataLoader(
        NeuDataset(test_samples(test, seen), label_map, eval_t),
        batch_size=cfg.batch, num_workers=cfg.workers,
    )

    model = FSCILModel(cfg, n_base=len(cfg.base_classes))
    model = train_base(model, base_loader, base_test_loader, cfg)

    model.freeze_backbone()
    a0 = evaluate(model, base_test_loader, cfg.device)
    session_accs = [a0]
    print(f"\nBase session done. acc: {a0:.3f}\n")

    # ---- Incremental sessions: imprint one novel class at a time ----
    for i, support in enumerate(inc_sessions, start=1):
        sup_loader = DataLoader(NeuDataset(support, label_map, eval_t),
                                batch_size=cfg.k_shot, num_workers=2)
        imprint_new_class(model, sup_loader, cfg.device)

        seen.append(cfg.novel_classes[i - 1])
        test_loader = DataLoader(
            NeuDataset(test_samples(test, seen), label_map, eval_t),
            batch_size=cfg.batch, num_workers=cfg.workers,
        )
        acc = evaluate(model, test_loader, cfg.device, n_classes=len(seen))
        session_accs.append(acc)
        print(f"[Session {i}] +{cfg.novel_classes[i - 1]} | "
              f"seen={len(seen)} classes | overall_acc={acc:.3f}")

    # ---- Final metrics ----
    n_seen = len(seen)
    base_test_loader_final = DataLoader(
        NeuDataset(test_samples(test, cfg.base_classes), label_map, eval_t),
        batch_size=cfg.batch, num_workers=cfg.workers,
    )
    novel_test_loader_final = DataLoader(
        NeuDataset(test_samples(test, cfg.novel_classes), label_map, eval_t),
        batch_size=cfg.batch, num_workers=cfg.workers,
    )
    base_acc_final  = evaluate(model, base_test_loader_final,  cfg.device, n_classes=n_seen)
    novel_acc_final = evaluate(model, novel_test_loader_final, cfg.device, n_classes=n_seen)
    compute_metrics(session_accs, base_acc_final, novel_acc_final)


if __name__ == "__main__":
    main()
