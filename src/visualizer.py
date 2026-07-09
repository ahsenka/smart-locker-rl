import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.environment import LockerEnvironment


class Visualizer:
    def __init__(self):
        os.makedirs("outputs/plots", exist_ok=True)
        os.makedirs("outputs/gifs", exist_ok=True)

    def _moving_average(self, values, window=100):
        averages = []
        for index in range(len(values)):
            start = max(0, index - window)
            window_values = values[start:index + 1]
            averages.append(sum(window_values) / len(window_values))
        return averages

    def reward_plot(self, total_rewards):
        plt.figure(figsize=(12, 6))
        plt.plot(total_rewards, color="#2f6f9f", linewidth=1)
        plt.title("Episode Based Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/plots/odul_grafigi.png", dpi=140)
        plt.close()

    def moving_average_plot(self, total_rewards):
        plt.figure(figsize=(12, 6))
        plt.plot(
            self._moving_average(total_rewards),
            color="#1b7f5f",
            linewidth=2,
        )
        plt.title("Moving Average Reward")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/plots/moving_average.png", dpi=140)
        plt.close()

    def success_plot(self, success_values):
        plt.figure(figsize=(12, 6))
        plt.plot(
            self._moving_average(success_values),
            color="#6f4aa3",
            linewidth=2,
        )
        plt.title("Success Rate")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/plots/basari_grafigi.png", dpi=140)
        plt.close()

    def acceptance_plot(self, acceptance_rates, invalid_move_rates):
        plt.figure(figsize=(12, 6))
        plt.plot(
            self._moving_average(acceptance_rates),
            label="Accepted package rate",
            color="#1b7f5f",
            linewidth=2,
        )
        plt.plot(
            self._moving_average(invalid_move_rates),
            label="Invalid move rate",
            color="#b94040",
            linewidth=2,
        )
        plt.title("Placement Quality During Training")
        plt.xlabel("Episode")
        plt.ylabel("Rate")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/plots/kabul_hata_orani.png", dpi=140)
        plt.close()

    def q_table_plot(self, q_table_sizes):
        plt.figure(figsize=(12, 6))
        plt.plot(q_table_sizes, color="#8c6d1f", linewidth=2)
        plt.title("Q-Table Growth")
        plt.xlabel("Episode")
        plt.ylabel("Visited State Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/plots/q_table_buyumesi.png", dpi=140)
        plt.close()

    def comparison_plot(self, baseline_results):
        names = list(baseline_results.keys())
        averages = [
            baseline_results[name]["average_reward"]
            for name in names
        ]

        plt.figure(figsize=(10, 6))
        colors = ["#1b7f5f", "#2f6f9f", "#b94040"]
        bars = plt.bar(names, averages, color=colors[:len(names)])
        plt.title("Agent Performance Comparison")
        plt.xlabel("Agent")
        plt.ylabel("Average Reward")
        plt.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, averages):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig("outputs/plots/agent_karsilastirma.png", dpi=140)
        plt.close()

    def comparison_timeseries_plot(self, baseline_results):
        plt.figure(figsize=(12, 6))
        colors = {
            "Q-Learning Agent": "#1b7f5f",
            "Rule-Based Agent": "#2f6f9f",
            "Random Agent": "#b94040",
        }

        for name, result in baseline_results.items():
            plt.plot(
                self._moving_average(result["rewards"], window=50),
                label=name,
                linewidth=2,
                color=colors.get(name),
            )

        plt.title("Q-Learning vs Baseline Agents")
        plt.xlabel("Evaluation Episode")
        plt.ylabel("Moving Average Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/plots/q_vs_baselines.png", dpi=140)
        plt.close()

        # Compatibility with the previous README image path.
        plt.figure(figsize=(12, 6))
        for name, result in baseline_results.items():
            plt.plot(
                self._moving_average(result["rewards"], window=50),
                label=name,
                linewidth=2,
                color=colors.get(name),
            )
        plt.title("Q-Learning Agent vs Random and Rule-Based Agents")
        plt.xlabel("Evaluation Episode")
        plt.ylabel("Moving Average Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("outputs/plots/q_vs_random.png", dpi=140)
        plt.close()

    def draw_simulation_frame(
        self,
        empty_lockers,
        step,
        package_label,
        action_name,
        reward,
        total_reward,
        accepted,
        filename,
    ):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_xlim(0, 7.5)
        ax.set_ylim(0, 5.8)
        ax.axis("off")

        title_color = "#1b7f5f" if accepted else "#b94040"
        ax.text(
            0.1,
            5.45,
            f"Smart Locker Simulation - Step {step + 1}",
            fontsize=16,
            fontweight="bold",
            color="#1f2933",
        )
        ax.text(
            0.1,
            5.05,
            f"Package: {package_label}",
            fontsize=11,
            color="#1f2933",
        )
        ax.text(
            0.1,
            4.72,
            f"Decision: {action_name} | Reward: {reward} | Total: {total_reward}",
            fontsize=11,
            color=title_color,
            fontweight="bold",
        )

        locker_rows = [
            (0, "Small", 4.0, "#7fb3d5"),
            (1, "Medium", 3.2, "#76d7c4"),
            (2, "Large", 2.4, "#f7dc6f"),
            (3, "XL", 1.6, "#f0b27a"),
            (4, "Cold", 0.8, "#85c1e9"),
        ]

        for action, label, y, color in locker_rows:
            total = LockerEnvironment().capacity[action]
            empty = empty_lockers[action]
            occupied = total - empty

            ax.text(0.1, y + 0.23, label, fontsize=11, fontweight="bold")

            for index in range(total):
                filled = index < occupied
                fill_color = color if filled else "#e5e7eb"
                edge_color = "#111827"
                rect = plt.Rectangle(
                    (1.1 + index * 0.85, y),
                    0.68,
                    0.52,
                    facecolor=fill_color,
                    edgecolor=edge_color,
                    linewidth=1,
                )
                ax.add_patch(rect)

        status = "Accepted" if accepted else "Rejected"
        ax.text(
            5.8,
            4.72,
            status,
            fontsize=13,
            fontweight="bold",
            color=title_color,
            ha="center",
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=130)
        plt.close()

    def create_simulation_gif(self, frame_paths, output_path):
        frames = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(output_path, frames, duration=900)

    def create_3d_frame(
        self,
        empty_lockers,
        step,
        package_label,
        action_name,
        reward,
        total_reward,
        filename,
    ):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        colors = ["#7fb3d5", "#76d7c4", "#f7dc6f", "#f0b27a", "#85c1e9"]
        labels = ["Small", "Medium", "Large", "XL", "Cold"]

        for action in range(5):
            total = LockerEnvironment().capacity[action]
            occupied = total - empty_lockers[action]
            for slot in range(total):
                z = slot * 0.42
                color = colors[action] if slot < occupied else "#d9dee5"
                alpha = 0.95 if slot < occupied else 0.35
                ax.bar3d(
                    action * 1.25,
                    0,
                    z,
                    0.75,
                    0.75,
                    0.34,
                    shade=True,
                    color=color,
                    alpha=alpha,
                    edgecolor="#263238",
                    linewidth=0.4,
                )

        ax.set_xticks(np.arange(5) * 1.25 + 0.35)
        ax.set_xticklabels(labels)
        ax.set_yticks([])
        ax.set_zlabel("Slots")
        ax.set_title(
            f"3D Locker Simulation - Step {step + 1}\n"
            f"{package_label} -> {action_name} | Reward {reward} | Total {total_reward}",
            pad=18,
        )
        ax.view_init(elev=24, azim=-55)
        ax.set_box_aspect((6, 1.8, 3))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88)
        plt.savefig(filename, dpi=130)
        plt.close()

    def create_summary_report(self, baseline_results):
        lines = [
            "Agent,Average Reward,Acceptance Rate,Invalid Move Rate",
        ]
        for name, result in baseline_results.items():
            lines.append(
                f"{name},"
                f"{result['average_reward']:.2f},"
                f"{result['acceptance_rate']:.3f},"
                f"{result['invalid_move_rate']:.3f}"
            )

        with open("outputs/agent_summary.csv", "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

    # Backward-compatible Turkish method names.
    def odul_grafigi_ciz(self, toplam_oduller):
        self.reward_plot(toplam_oduller)

    def moving_average_grafigi(self, toplam_oduller):
        self.moving_average_plot(toplam_oduller)

    def basari_grafigi(self, basari_listesi):
        self.success_plot(basari_listesi)

    def karsilastirma_grafigi(self, q_oduller, random_oduller):
        baseline_results = {
            "Q-Learning Agent": {"rewards": q_oduller, "average_reward": sum(q_oduller) / len(q_oduller)},
            "Random Agent": {"rewards": random_oduller, "average_reward": sum(random_oduller) / len(random_oduller)},
        }
        self.comparison_timeseries_plot(baseline_results)

    def gif_olustur(self, frame_listesi):
        self.create_simulation_gif(frame_listesi, "outputs/gifs/locker_simulasyonu.gif")
