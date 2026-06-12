from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def exponential_moving_average(values: pd.Series, alpha: float = 0.1) -> np.ndarray:
    if values.empty:
        return np.asarray([], dtype=float)
    raw = pd.to_numeric(values, errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
    smoothed = [raw[0]]
    for value in raw[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return np.asarray(smoothed, dtype=float)


def normalize_rewards_frame(rewards_df: pd.DataFrame) -> pd.DataFrame:
    df = rewards_df.copy()
    if "Episode" not in df.columns:
        df["Episode"] = np.arange(len(df))
    if "Raw_Reward" not in df.columns:
        if "Reward" in df.columns:
            df["Raw_Reward"] = df["Reward"]
        elif df.shape[1] >= 2:
            df["Raw_Reward"] = df.iloc[:, 1]
        else:
            raise ValueError("Reward CSV must contain Raw_Reward, Reward, or at least two columns.")
    if "Smoothed_Reward" not in df.columns:
        df["Smoothed_Reward"] = exponential_moving_average(df["Raw_Reward"])
    return df


def normalize_loss_frame(loss_df: pd.DataFrame) -> pd.DataFrame:
    df = loss_df.copy()
    if df.empty:
        return df
    if "Episode" not in df.columns:
        df["Episode"] = np.arange(len(df))
    return df


def plot_reward_loss_curves(
    rewards_df: pd.DataFrame,
    loss_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    rewards_df = normalize_rewards_frame(rewards_df)
    loss_df = normalize_loss_frame(loss_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    axes[0].plot(
        rewards_df["Episode"],
        rewards_df["Raw_Reward"],
        color="#4C78A8",
        alpha=0.28,
        linewidth=1.0,
        label="Raw reward",
    )
    axes[0].plot(
        rewards_df["Episode"],
        rewards_df["Smoothed_Reward"],
        color="#D84A4A",
        linewidth=2.0,
        label="Smoothed reward",
    )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward curve")
    axes[0].grid(True, color="#DADADA", linewidth=0.65, alpha=0.7)
    axes[0].legend(frameon=True, fancybox=False, edgecolor="#BDBDBD")

    if loss_df.empty:
        axes[1].text(0.5, 0.5, "No loss records", ha="center", va="center", transform=axes[1].transAxes)
    else:
        if "Actor_Loss" in loss_df.columns:
            axes[1].plot(loss_df["Episode"], loss_df["Actor_Loss"], color="#2F6BFF", label="Actor loss")
        if "Critic_Loss" in loss_df.columns:
            axes[1].plot(loss_df["Episode"], loss_df["Critic_Loss"], color="#E6862E", label="Critic loss")
        if "Total_Loss" in loss_df.columns:
            axes[1].plot(loss_df["Episode"], loss_df["Total_Loss"], color="#6A3D9A", label="Total loss")
        if "Entropy" in loss_df.columns:
            ax2 = axes[1].twinx()
            ax2.plot(loss_df["Episode"], loss_df["Entropy"], color="#3CB44B", alpha=0.65, label="Entropy")
            ax2.set_ylabel("Entropy")
            handles1, labels1 = axes[1].get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            axes[1].legend(handles1 + handles2, labels1 + labels2, frameon=True, fancybox=False, edgecolor="#BDBDBD")
        else:
            axes[1].legend(frameon=True, fancybox=False, edgecolor="#BDBDBD")

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss curve")
    axes[1].grid(True, color="#DADADA", linewidth=0.65, alpha=0.7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", direction="out", length=3.2, width=0.8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(rewards_df: pd.DataFrame, loss_df: pd.DataFrame, output_path: Path) -> None:
    """Backward-compatible wrapper used by train.py."""
    plot_reward_loss_curves(rewards_df, loss_df, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize reward and loss curves.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory containing rewards/loss CSV files.")
    parser.add_argument("--rewards-csv", type=Path, default=None, help="Explicit rewards CSV path.")
    parser.add_argument("--loss-csv", type=Path, default=None, help="Explicit loss CSV path.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path. A PDF is also saved.")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir is None and args.rewards_csv is None:
        raise SystemExit("Pass --run-dir or --rewards-csv.")

    rewards_csv = args.rewards_csv or args.run_dir / "rewards_history.csv"
    loss_csv = args.loss_csv or args.run_dir / "loss_history.csv"
    output = args.output or (args.run_dir / "reward_loss_curves.png")

    rewards = pd.read_csv(rewards_csv)
    losses = pd.read_csv(loss_csv) if loss_csv.exists() else pd.DataFrame()
    plot_reward_loss_curves(rewards, losses, output, title=args.title)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
