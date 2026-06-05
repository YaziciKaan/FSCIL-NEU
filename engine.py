import torch
import torch.nn as nn
import torch.nn.functional as F



def train_base(model, train_loader, test_loader, cfg):
    device = cfg.device
    model.to(device)
    optim = torch.optim.SGD(
        model.parameters(), lr=cfg.base_lr, momentum=0.9,
        weight_decay=cfg.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.base_epoch)
    ce = nn.CrossEntropyLoss()

    for epoch in range(cfg.base_epoch):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = ce(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            run_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()
            total += x.size(0)
        scheduler.step()

        te_acc = evaluate(model, test_loader, device)
        print(f"[Base] epoch {epoch +1:>3}/{cfg.base_epoch} | "
              f"loss {run_loss / total:.4f} | "
              f"train_acc {correct / total:.3f} | test_acc {te_acc:.3f}")
    return model

@torch.no_grad()
def evaluate(model, loader, device, n_classes=None):
    model.eval()
    correct , total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        if n_classes is not None:
            output = output[:, :n_classes]
        correct += (output.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct/max(total, 1)



def compute_metrics(session_accs: list, base_acc_final: float, novel_acc_final: float):
    avg_acc = sum(session_accs) / len(session_accs)
    pd      = session_accs[0] - session_accs[-1]
    hm      = (2 * base_acc_final * novel_acc_final /
               (base_acc_final + novel_acc_final + 1e-8))
    print(f"\n{'='*45}")
    print(f"  Per-session acc : {[f'{a:.3f}' for a in session_accs]}")
    print(f"  Average acc     : {avg_acc:.3f}")
    print(f"  Perf. Dropping  : {pd:.3f}")
    print(f"  Base acc (final): {base_acc_final:.3f}")
    print(f"  Novel acc(final): {novel_acc_final:.3f}")
    print(f"  Harmonic Mean   : {hm:.3f}")
    print(f"{'='*45}\n")


@torch.no_grad()
def imprint_new_class(model, support_loader, device):
    model.eval()
    feats = [model.encode(x.to(device)) for x, _ in support_loader]
    feats = torch.cat(feats, dim=0) # (K, 512)
    proto = F.normalize(feats, dim=1).mean(0, keepdim=True)
    model.head.append_prototypes(proto)