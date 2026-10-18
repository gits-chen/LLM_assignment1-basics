from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.utils import (
    evaluate,
    parse_args,
    save_config,
    validate_resume_config,
)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    project_root = Path(__file__).resolve().parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = args.batch_size
    context_length = args.context_length
    total_steps = args.total_steps
    
    dataset = np.memmap(
        project_root / "artifacts/encoded/tinystories_train.bin",
        dtype="<u2",
        mode="r",
    )
    valid_dataset = np.memmap(
        project_root / "artifacts/encoded/tinystories_valid.bin",
        dtype="<u2",
        mode="r",
    )

    model = TransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(device),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    start_step = 0

    if args.resume is not None:
        resume_path = project_root / args.resume
        validate_resume_config(resume_path, args)

        start_step = load_checkpoint(
            resume_path,
            model,
            optimizer,
        )

        print(f"已恢复检查点，已完成 {start_step} 步")

    if total_steps <= start_step:
        raise ValueError(
            "训练总步数必须大于检查点中已完成的步数"
        )

    model.train()
    print(f"训练设备：{device}")
    initial_valid_loss = evaluate(
        model,
        valid_dataset,
        batch_size,
        context_length,
        device,
    )
    print(
        f"step={start_step} "
        f"valid_loss={initial_valid_loss:.4f}"
    )

    checkpoint_path = project_root / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)


    for step in range(start_step, total_steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.lr_decay_steps,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        input_ids, target_ids = get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = cross_entropy(logits, target_ids)

        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()

        print(
            f"step={step + 1}/{total_steps} "
            f"loss={loss.item():.4f} "
            f"lr={lr:.6f}"
        )

        if (
            (step + 1) % args.eval_every == 0
            or step + 1 == total_steps
        ):
            valid_loss = evaluate(
                model,
                valid_dataset,
                batch_size,
                context_length,
                device,
            )

            print(
                f"step={step + 1} "
                f"valid_loss={valid_loss:.4f}"
            )

        completed_steps = step + 1

        if (
            completed_steps % args.save_every == 0
            and completed_steps < total_steps
        ):
            periodic_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}"
                f"_step_{completed_steps}"
                f"{checkpoint_path.suffix}"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=completed_steps,
                out=periodic_path,
            )

            save_config(periodic_path, args)

            print(f"中途检查点已保存：{periodic_path}")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_steps,
        out=checkpoint_path,
    )

    save_config(checkpoint_path, args)

    print(f"检查点已保存：{checkpoint_path}")


if __name__ == "__main__":
    main()
