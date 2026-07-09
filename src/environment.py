import random


class LockerEnvironment:
    """Smart locker placement environment with richer package constraints."""

    SIZE_NAMES = {
        0: "small",
        1: "medium",
        2: "large",
        3: "xlarge",
    }

    PRIORITY_NAMES = {
        0: "standard",
        1: "express",
        2: "urgent",
    }

    ACTION_NAMES = {
        0: "small locker",
        1: "medium locker",
        2: "large locker",
        3: "xlarge locker",
        4: "cold locker",
    }

    def __init__(self, scenario="balanced", seed=None):
        self.scenario = scenario
        self.random = random.Random(seed)

        self.capacity = {
            0: 6,
            1: 5,
            2: 4,
            3: 3,
            4: 3,
        }

        self.reset()

    def reset(self):
        self.empty_lockers = dict(self.capacity)
        self.last_info = {}
        self.current_package = self._generate_package()
        return self.get_state()

    def _weighted_choice(self, values, weights):
        return self.random.choices(values, weights=weights, k=1)[0]

    def _generate_package(self):
        if self.scenario == "peak_day":
            size_weights = [0.35, 0.30, 0.22, 0.13]
            priority_weights = [0.45, 0.35, 0.20]
            cold_probability = 0.18
            fragile_probability = 0.25
        elif self.scenario == "cold_chain":
            size_weights = [0.42, 0.28, 0.20, 0.10]
            priority_weights = [0.50, 0.30, 0.20]
            cold_probability = 0.42
            fragile_probability = 0.22
        else:
            size_weights = [0.48, 0.28, 0.17, 0.07]
            priority_weights = [0.60, 0.28, 0.12]
            cold_probability = 0.16
            fragile_probability = 0.18

        size = self._weighted_choice([0, 1, 2, 3], size_weights)
        priority = self._weighted_choice([0, 1, 2], priority_weights)
        cold_required = int(self.random.random() < cold_probability)
        fragile = int(self.random.random() < fragile_probability)

        return {
            "size": size,
            "priority": priority,
            "cold_required": cold_required,
            "fragile": fragile,
        }

    def get_state(self):
        package = self.current_package
        return (
            self.empty_lockers[0],
            self.empty_lockers[1],
            self.empty_lockers[2],
            self.empty_lockers[3],
            self.empty_lockers[4],
            package["size"],
            package["priority"],
            package["cold_required"],
            package["fragile"],
        )

    def get_package_label(self, package=None):
        package = package or self.current_package
        labels = [
            self.SIZE_NAMES[package["size"]],
            self.PRIORITY_NAMES[package["priority"]],
        ]

        if package["cold_required"]:
            labels.append("cold chain")
        if package["fragile"]:
            labels.append("fragile")

        return ", ".join(labels)

    def is_action_feasible(self, action):
        package = self.current_package

        if action not in self.empty_lockers:
            return False

        if self.empty_lockers[action] <= 0:
            return False

        if action == 4:
            return package["cold_required"] and package["size"] <= 2

        return action >= package["size"]

    def valid_actions(self):
        actions = [
            action
            for action in self.empty_lockers
            if self.is_action_feasible(action)
        ]

        if self.current_package["cold_required"] and 4 in actions:
            actions.remove(4)
            actions.insert(0, 4)

        return actions

    @staticmethod
    def valid_actions_from_state(state):
        (
            empty_small,
            empty_medium,
            empty_large,
            empty_xlarge,
            empty_cold,
            package_size,
            _priority,
            cold_required,
            _fragile,
        ) = state

        empty = {
            0: empty_small,
            1: empty_medium,
            2: empty_large,
            3: empty_xlarge,
            4: empty_cold,
        }

        valid = []
        for action in range(package_size, 4):
            if empty[action] > 0:
                valid.append(action)

        if cold_required and package_size <= 2 and empty[4] > 0:
            valid.insert(0, 4)

        return valid

    def _placement_reward(self, action):
        package = self.current_package
        size = package["size"]
        priority = package["priority"]
        cold_required = package["cold_required"]
        fragile = package["fragile"]

        if action == 4:
            reward = 13
            if cold_required:
                reward += 10
            if priority == 2:
                reward += 2
            if fragile:
                reward += 2
            return reward

        size_gap = action - size
        if size_gap == 0:
            reward = 12
        elif size_gap == 1:
            reward = 5
        else:
            reward = 1

        if size <= 1 and action >= 2:
            reward -= 5
        if size <= 2 and action == 3:
            reward -= 3
        if cold_required:
            reward -= 18
        if priority == 2 and action <= 1:
            reward -= 2
        if fragile and action == size:
            reward += 2

        return reward

    def step(self, action):
        done = False
        package_before_action = dict(self.current_package)

        if not self.is_action_feasible(action):
            reward = -35 if self.current_package["cold_required"] else -25
            self.last_info = {
                "accepted": False,
                "action": action,
                "reward": reward,
                "package": package_before_action,
                "reason": "invalid placement",
            }
            return self.get_state(), reward, done

        self.empty_lockers[action] -= 1
        reward = self._placement_reward(action)

        if sum(self.empty_lockers.values()) == 0:
            done = True

        self.last_info = {
            "accepted": True,
            "action": action,
            "reward": reward,
            "package": package_before_action,
            "reason": "placed",
        }

        if not done:
            self.current_package = self._generate_package()

        return self.get_state(), reward, done
