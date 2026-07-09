from src.environment import LockerEnvironment
from src.trainer import Trainer
from src.visualizer import Visualizer


def run_simulation(env, agent, visualizer, max_steps=18):
    state = env.reset()
    frame_paths = []
    frame_paths_3d = []
    total_test_reward = 0

    print("\nsmart locker simulation started\n")

    for step in range(max_steps):
        # Egitilmis ajanla test simulasyonu yapiliyor
        action = agent.select_action(
            state,
            training=False,
            valid_actions=env.valid_actions(),
        )
        package_label = env.get_package_label()
        next_state, reward, done = env.step(action)
        total_test_reward += reward

        accepted = env.last_info["accepted"]
        action_name = LockerEnvironment.ACTION_NAMES.get(action, "unknown")

        frame_path = f"outputs/gifs/frame_{step}.png"
        frame_path_3d = f"outputs/gifs/frame_3d_{step}.png"

        # Hem 2D hem de 3D frame uretiliyor
        visualizer.draw_simulation_frame(
            env.empty_lockers,
            step,
            package_label,
            action_name,
            reward,
            total_test_reward,
            accepted,
            frame_path,
        )
        visualizer.create_3d_frame(
            env.empty_lockers,
            step,
            package_label,
            action_name,
            reward,
            total_test_reward,
            frame_path_3d,
        )

        frame_paths.append(frame_path)
        frame_paths_3d.append(frame_path_3d)

        print(f"step: {step + 1}")
        print(f"state: {state}")
        print(f"package: {package_label}")
        print(f"selected action: {action_name}")
        print(f"reward: {reward}")
        print(f"accepted: {accepted}")
        print("-----------------------")

        state = next_state

        if done:
            print("all lockers are full")
            break

    visualizer.create_simulation_gif(
        frame_paths,
        "outputs/gifs/locker_simulasyonu.gif",
    )
    visualizer.create_simulation_gif(
        frame_paths_3d,
        "outputs/gifs/locker_simulasyonu_3d.gif",
    )

    print("\nsimulation gifs created\n")


def main():
    trainer = Trainer(scenario="balanced", seed=42)
    visualizer = Visualizer()

    # Once egitim ve karsilastirma sonuclari aliniyor
    agent, total_rewards = trainer.train()
    baseline_results = trainer.evaluate_baselines()

    # Grafikler outputs/plots klasorune kaydediliyor
    visualizer.reward_plot(total_rewards)
    visualizer.success_plot(trainer.successful_episodes)
    visualizer.moving_average_plot(total_rewards)
    visualizer.acceptance_plot(
        trainer.acceptance_rates,
        trainer.invalid_move_rates,
    )
    visualizer.q_table_plot(trainer.q_table_sizes)
    visualizer.comparison_plot(baseline_results)
    visualizer.comparison_timeseries_plot(baseline_results)
    visualizer.create_summary_report(baseline_results)

    simulation_env = LockerEnvironment(scenario="balanced", seed=84)
    run_simulation(simulation_env, agent, visualizer)

    print("\nall processes completed\n")


if __name__ == "__main__":
    main()
