import os
import runpy
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import optuna
import wandb
from rich import print
from tensorboard.backend.event_processing import event_accumulator


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Tuner:
    def __init__(
        self,
        script: str,
        metric: str,
        target_scores: Dict[str, Optional[List[float]]],
        params_fn: Callable[[optuna.Trial], Dict],
        direction: str = "maximize",
        aggregation_type: str = "average",
        metric_last_n_average_window: int = 50,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        storage: str = "sqlite:///cleanrl_hpopt.db",
        study_name: str = "",
        wandb_kwargs: Dict[str, any] = {},
    ) -> None:
        self.script = script
        self.metric = metric
        self.target_scores = target_scores
        if len(self.target_scores) > 1:
            if None in self.target_scores.values():
                raise ValueError(
                    "If there are multiple environments, the target scores must be specified for each environment."
                )

        self.params_fn = params_fn
        self.direction = direction
        self.aggregation_type = aggregation_type
        if self.aggregation_type == "average":
            self.aggregation_fn = np.average
        elif self.aggregation_type == "median":
            self.aggregation_fn = np.median
        elif self.aggregation_type == "max":
            self.aggregation_fn = np.max
        elif self.aggregation_type == "min":
            self.aggregation_fn = np.min
        else:
            raise ValueError(f"Unknown aggregation type {self.aggregation_type}")
        self.metric_last_n_average_window = metric_last_n_average_window
        self.pruner = pruner
        self.sampler = sampler
        self.storage = storage
        self.study_name = study_name
        if len(self.study_name) == 0:
            self.study_name = f"tuner_{int(time.time())}"
        self.wandb_kwargs = wandb_kwargs

    def tune(self, num_trials: int, num_seeds: int) -> None:
        def objective(trial: optuna.Trial):
            params = self.params_fn(trial)
            run = None
            if len(self.wandb_kwargs.keys()) > 0:
                run = wandb.init(
                    **self.wandb_kwargs,
                    config=params,
                    name=f"{self.study_name}_{trial.number}",
                    group=self.study_name,
                    save_code=True,
                    reinit=True,
                )

            algo_command = []
            for key, value in params.items():
                if isinstance(value, bool):
                    if value:
                        algo_command.append(f"--{key}")
                    else:
                        algo_command.append(f"--no-{key}")
                else:
                    algo_command.append(f"--{key}={value}")
            
            normalized_scoress = []
            for seed in range(num_seeds):
                normalized_scores = []
                for env_id in self.target_scores.keys():
                    sys.argv = algo_command + [f"--env-id={env_id}", f"--seed={seed}"]
                    with HiddenPrints():
                        experiment = runpy.run_path(path_name=self.script, run_name="__main__")

                    # read metric from tensorboard
                    ea = event_accumulator.EventAccumulator(f"runs/{experiment['run_name']}")
                    ea.Reload()
                    metric_values = [
                        scalar_event.value for scalar_event in ea.Scalars(self.metric)[-self.metric_last_n_average_window :]
                    ]
                    print(
                        f"The average episodic return on {env_id} is {np.average(metric_values)} averaged over the last {self.metric_last_n_average_window} episodes."
                    )
                    if self.target_scores[env_id] is not None:
                        normalized_scores += [
                            (np.average(metric_values) - self.target_scores[env_id][0])
                            / (self.target_scores[env_id][1] - self.target_scores[env_id][0])
                        ]
                    else:
                        normalized_scores += [np.average(metric_values)]
                    if run:
                        run.log({f"{env_id}_return": np.average(metric_values)})

                normalized_scoress += [normalized_scores]
                aggregated_normalized_score = self.aggregation_fn(normalized_scores)
                print(f"The {self.aggregation_type} normalized score is {aggregated_normalized_score} with num_seeds={seed}")
                trial.report(aggregated_normalized_score, step=seed)
                if run:
                    run.log({"aggregated_normalized_score": aggregated_normalized_score})
                if trial.should_prune():
                    if run:
                        run.finish(quiet=True)
                    raise optuna.TrialPruned()

            if run:
                run.finish(quiet=True)
            return np.average(
                self.aggregation_fn(normalized_scoress, axis=1)
            )  # we alaways return the average of the aggregated normalized scores

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            pruner=self.pruner,
            sampler=self.sampler,
        )
        print("==========================================================================================")
        print("run another tuner with the following command:")
        print(f"python -m cleanrl_utils.tuner --study-name {self.study_name}")
        print("==========================================================================================")
        study.optimize(
            objective,
            n_trials=num_trials,
        )
        print(f"The best trial obtains a normalized score of {study.best_trial.value}", study.best_trial.params)
        return study.best_trial
    
# no folds 171.87724493026735 {'learning-rate': 0.0017498971793561953, 'num-minibatches': 2, 'update-epochs': 8, 'num-steps': 16, 'vf-coef': 1.4863089240144056, 'max-grad-norm': 4.833840924269445}

# folds 10x lr The best trial obtains a normalized score of 120.10150365511576
# {'learning-rate': 0.000557976639996598, 'num-minibatches': 4, 'update-epochs': 2, 'num-steps': 5, 'vf-coef': 3.0254739122093386, 'max-grad-norm': 4.633703613193437}

# folds same lr The best trial obtains a normalized score of 122.49960039556028
# {'learning-rate': 0.0018918314677542124, 'num-minibatches': 4, 'update-epochs': 8, 'num-steps': 32, 'vf-coef': 4.39832402246543, 'max-grad-norm': 1.6532482357812137}