from itertools import cycle
from types import SimpleNamespace

import torch

from ript.algos.rl_optimizers.rl_optimizer import RLOptimizer
from ript.algos.rl_optimizers.rollout_generator import RolloutBudgetExhausted, RolloutGenerator


class FakeEnv:
    def close(self):
        pass


class FakeRunner:
    def __init__(self, scenario):
        self.scenario = scenario
        self.calls = 0

    def create_env(self, task_name):
        return FakeEnv(), 0, 1

    def run_policy_in_env(self, task_name, model, env_init_states, render=False, created_env=None, random_init=False):
        scenario = self.scenario
        if scenario == "mixed_then_zero" and self.calls > 0:
            scenario = "all_zero"
        self.calls += 1
        for index in range(4):
            if scenario == "all_zero":
                success = False
            elif scenario == "all_one":
                success = True
            elif scenario in ("mixed", "mixed_then_zero"):
                success = index % 2 == 0
            else:
                raise ValueError(scenario)
            yield success, float(success), {
                "context_tokens": [0],
                "action_indices": [0],
                "policy_inference_steps": [1],
            }


class DummyReward:
    def compute_reward(self, index, episode, batch):
        return 0.0


def make_batch():
    return {
        "task_id": torch.tensor([0, 0]),
        "init_state": {
            "states": torch.ones(2, 1, 4),
            "pad_mask": torch.ones(2, 4, dtype=torch.bool),
        },
    }


def make_generator(scenario, early_stop=1.0):
    return RolloutGenerator(
        rloo_batch_size=4,
        demo_batch_size=2,
        early_stop_percentage=early_stop,
        enable_dynamic_sampling=True,
        task_names_to_use=["task"],
        env_runner=FakeRunner(scenario),
        max_samples_checked=3,
    )


def generate(generator):
    model = SimpleNamespace(device=torch.device("cpu"))
    return generator.generate_rollouts(model, make_batch(), cycle([make_batch()]), [make_batch()])


def check_uniform(scenario, counter_key):
    generator = make_generator(scenario)
    optimizer = RLOptimizer(generator, DummyReward())
    interface = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
    metrics = optimizer.optimize(
        interface,
        make_batch(),
        optimizers=[],
        data_iterator=cycle([make_batch()]),
        dataloader=[make_batch()],
    )
    assert metrics["skipped_no_advantage"] == 1
    assert metrics["rollout_budget_exhausted"] == 1
    assert metrics["termination_reason"] == "max_samples_checked"
    assert metrics["samples_checked"] == 3
    assert metrics["accepted_groups"] == 0
    assert metrics[counter_key] == 3
    assert metrics["discarded_episodes"] == 12
    generator.cleanup()


def check_mixed():
    generator = make_generator("mixed")
    episodes, task_ids, valid_mask, checked = generate(generator)
    assert len(episodes) == 8
    assert checked == 2
    assert valid_mask.all()
    assert generator.last_rollout_stats["accepted_groups"] == 2
    assert generator.last_rollout_stats["accepted_episodes"] == 8
    assert generator.last_rollout_stats["discarded_episodes"] == 0
    generator.cleanup()


def check_partial_budget():
    generator = make_generator("mixed_then_zero")
    try:
        generate(generator)
    except RolloutBudgetExhausted as exc:
        assert exc.stats["accepted_groups"] == 1
        assert exc.stats["accepted_episodes"] == 4
        assert exc.stats["discarded_episodes"] == 8
        assert exc.stats["termination_reason"] == "max_samples_checked"
    else:
        raise AssertionError("partial budget did not raise")
    generator.cleanup()


def check_early_stop():
    generator = make_generator("mixed", early_stop=0.5)
    episodes, task_ids, valid_mask, checked = generate(generator)
    assert len(episodes) == 8
    assert checked == 1
    assert valid_mask.tolist() == [True, True, True, True, False, False, False, False]
    assert generator.last_rollout_stats["termination_reason"] == "global_early_stop"
    generator.cleanup()


check_uniform("all_zero", "discarded_all_zero_groups")
check_uniform("all_one", "discarded_all_one_groups")
check_mixed()
check_partial_budget()
check_early_stop()
print("Bounded dynamic sampling validation passed")
