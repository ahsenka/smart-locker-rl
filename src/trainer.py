from src.agent import QLearningAgent, RandomAgent, RuleBasedAgent
from src.environment import LockerEnvironment


class Trainer:
    def __init__(self, scenario="balanced", seed=42):
        self.scenario = scenario
        self.seed = seed
        self.env = LockerEnvironment(scenario=scenario, seed=seed)
        self.agent = QLearningAgent(seed=seed)

        self.episode_count = 6000
        self.max_step = 40

        self.total_rewards = []
        self.successful_episodes = []
        self.acceptance_rates = []
        self.invalid_move_rates = []
        self.q_table_sizes = []

        self.baseline_results = {}

    def train(self):
        for episode in range(self.episode_count):
            state = self.env.reset()
            total_reward = 0
            accepted_count = 0
            invalid_count = 0

            for step in range(self.max_step):
                action = self.agent.select_action(
                    state,
                    training=True,
                    valid_actions=self.env.valid_actions(),
                )
                next_state, reward, done = self.env.step(action)

                self.agent.learn(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                state = next_state
                total_reward += reward

                if self.env.last_info.get("accepted"):
                    accepted_count += 1
                else:
                    invalid_count += 1

                if done:
                    break

            steps_taken = step + 1
            self.total_rewards.append(total_reward)
            self.successful_episodes.append(1 if total_reward > 0 else 0)
            self.acceptance_rates.append(accepted_count / steps_taken)
            self.invalid_move_rates.append(invalid_count / steps_taken)
            self.q_table_sizes.append(len(self.agent.q_table))
            self.agent.update_epsilon()

            if (episode + 1) % 500 == 0:
                print(
                    f"episode: {episode + 1} | "
                    f"total reward: {total_reward} | "
                    f"epsilon: {self.agent.epsilon:.3f} | "
                    f"q states: {len(self.agent.q_table)}"
                )

        print("\ntraining completed\n")
        return self.agent, self.total_rewards

    def evaluate_agent(self, agent, episodes=600, training=False):
        rewards = []
        acceptance_rates = []
        invalid_move_rates = []

        eval_env = LockerEnvironment(scenario=self.scenario, seed=self.seed + 100)

        for _ in range(episodes):
            state = eval_env.reset()
            total_reward = 0
            accepted_count = 0
            invalid_count = 0

            for step in range(self.max_step):
                valid_actions = (
                    eval_env.valid_actions()
                    if agent is self.agent
                    else None
                )
                action = agent.select_action(
                    state,
                    training=training,
                    valid_actions=valid_actions,
                )
                next_state, reward, done = eval_env.step(action)

                state = next_state
                total_reward += reward

                if eval_env.last_info.get("accepted"):
                    accepted_count += 1
                else:
                    invalid_count += 1

                if done:
                    break

            steps_taken = step + 1
            rewards.append(total_reward)
            acceptance_rates.append(accepted_count / steps_taken)
            invalid_move_rates.append(invalid_count / steps_taken)

        return {
            "rewards": rewards,
            "average_reward": sum(rewards) / len(rewards),
            "acceptance_rate": sum(acceptance_rates) / len(acceptance_rates),
            "invalid_move_rate": sum(invalid_move_rates) / len(invalid_move_rates),
        }

    def evaluate_baselines(self):
        q_evaluation_epsilon = self.agent.epsilon
        self.agent.epsilon = 0

        self.baseline_results = {
            "Q-Learning Agent": self.evaluate_agent(self.agent),
            "Rule-Based Agent": self.evaluate_agent(RuleBasedAgent()),
            "Random Agent": self.evaluate_agent(
                RandomAgent(seed=self.seed + 200),
            ),
        }

        self.agent.epsilon = q_evaluation_epsilon
        print("\nbaseline evaluation completed\n")
        return self.baseline_results

    # Backward-compatible Turkish method names.
    def egit(self):
        return self.train()

    def random_agent_test_et(self):
        result = self.evaluate_agent(RandomAgent(seed=self.seed + 200))
        return result["rewards"]

    @property
    def toplam_oduller(self):
        return self.total_rewards

    @property
    def basarili_episode(self):
        return self.successful_episodes
