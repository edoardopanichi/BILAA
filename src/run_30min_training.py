"""Run and evaluate the current best evolutionary-training configuration.

The run is intentionally separate from the notebook and from the historical
model store.  It saves NumPy weights and a JSON report in a timestamped
``.training_runs`` directory, so repeating the job does not overwrite an
existing model.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Select the backend before importing Keras or any project module that imports
# it.  This is also the backend used by the validated short benchmark.
os.environ.setdefault("KERAS_BACKEND", "torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import chess
import numpy as np

from chess_functions import stockfish_eng
from evaluation_class import evaluator, prediction_to_values
from evolutionary_algorithm import genetic_algorithm
from fitness_function import DEFAULT_OPENING_SEQUENCES, fitness
from monte_carlo_search_tree import MCTS


CONFIG = {
    "seed": 123,
    "population": 10,
    "generations": 40,
    "mcst_epochs": 3,
    "mcst_depth": 3,
    "max_plies": 60,
    "games_per_agent": 4,
    "state_features": True,
    "root_perspective": True,
    "terminal_reward": 5.0,
    "draw_material_weight": 0.1,
    "random_ties": False,
    "scoring": "robust",
    "evaluation_seed": 123,
    "opening_positions": len(DEFAULT_OPENING_SEQUENCES),
    "variation_mode": "enhanced",
    "mutation_gene_rate": 0.01,
    "mutation_scale": 0.1,
    "preserve_global_best": True,
    "architecture": "local",
    "stockfish_skill": 0,
    "stockfish_depth": 2,
    "stockfish_max_plies": 80,
    "stockfish_games": 2,
}


def _json_float(value):
    return float(value) if value is not None else None


def _save_weights(model, path: Path):
    weights = model.get_weights()
    np.savez_compressed(path, **{f"w{index}": value for index, value in enumerate(weights)})
    return [list(value.shape) for value in weights]


def _close_stockfish(wrapper):
    if wrapper is None:
        return
    engine = getattr(wrapper, "engine", None)
    close = getattr(engine, "send_quit_command", None)
    if callable(close):
        try:
            close()
        except Exception:
            # The evaluation result is still useful if the subprocess has
            # already exited.
            pass


def _play_fixed_stockfish_game(agent, stockfish_is_white, max_plies, skill):
    """Play exactly one bounded game and return its result from the EA view."""
    wrapper = stockfish_eng()
    wrapper.skill_value(skill)
    mcts = MCTS(state_features=True)
    model = agent.neural_network
    ea_color = chess.BLACK if stockfish_is_white else chess.WHITE
    board = chess.Board()
    plies = 0

    def evaluation(input_batch):
        batch = np.asarray(input_batch)
        if batch.ndim == 3:
            batch = batch[None, ...]
        return prediction_to_values(model(batch))

    try:
        while plies < max_plies and not board.is_game_over():
            if board.turn == ea_color:
                move, _ = mcts.simple_mcst(
                    board,
                    evaluation,
                    epochs=CONFIG["mcst_epochs"],
                    depth=CONFIG["mcst_depth"],
                    root_perspective=CONFIG["root_perspective"],
                    terminal_reward=CONFIG["terminal_reward"],
                    random_ties=CONFIG["random_ties"],
                )
                board.push(move)
            else:
                move = wrapper.play_best_move(board)
                try:
                    board.push_uci(move)
                except ValueError:
                    board.push_san(move)
            plies += 1

        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            result = "D"
        elif outcome.winner == ea_color:
            result = "W"
        else:
            result = "L"
        return {"result": result, "plies": plies, "fen": board.fen()}
    finally:
        _close_stockfish(wrapper)


def _evaluate_stockfish(agent):
    games = []
    for index in range(CONFIG["stockfish_games"]):
        stockfish_is_white = index % 2 == 0
        game = _play_fixed_stockfish_game(
            agent,
            stockfish_is_white=stockfish_is_white,
            max_plies=CONFIG["stockfish_max_plies"],
            skill=CONFIG["stockfish_skill"],
        )
        game["stockfish_is_white"] = stockfish_is_white
        games.append(game)
        print(
            "stockfish_game="
            f"{index + 1}/{CONFIG['stockfish_games']} "
            f"result={game['result']} plies={game['plies']}",
            flush=True,
        )
    return games


def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO_ROOT / ".training_runs" / f"ea_30min_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(CONFIG, indent=2) + "\n", encoding="utf-8"
    )

    print(f"run_dir={run_dir}", flush=True)
    print(f"config={json.dumps(CONFIG, sort_keys=True)}", flush=True)

    model = evaluator(include_state=True).simple_eval_model(
        architecture=CONFIG["architecture"]
    )
    generation_counter = [0]
    generation_metrics = []
    training_started = time.perf_counter()

    def training_fitness(agents, mcst_epochs, mcst_depth):
        generation_index = generation_counter[0]
        result_agents, decisive_games = fitness(
            agents,
            mcst_epochs,
            mcst_depth,
            max_plies=CONFIG["max_plies"],
            verbose=False,
            schedule="balanced",
            scoring=CONFIG["scoring"],
            games_per_agent=CONFIG["games_per_agent"],
            state_features=CONFIG["state_features"],
            root_perspective=CONFIG["root_perspective"],
            terminal_reward=CONFIG["terminal_reward"],
            draw_material_weight=CONFIG["draw_material_weight"],
            random_ties=CONFIG["random_ties"],
            opening_sequences=DEFAULT_OPENING_SEQUENCES,
            evaluation_seed=CONFIG["evaluation_seed"] + generation_index,
        )
        generation_counter[0] += 1
        metrics = dict(getattr(fitness, "last_metrics", {}))
        metrics["generation"] = generation_counter[0]
        generation_metrics.append(metrics)
        best = max(agent.fitness for agent in result_agents)
        elapsed = time.perf_counter() - training_started
        print(
            f"generation={generation_counter[0]}/{CONFIG['generations']} "
            f"best_fitness={best:.6f} decisive_games={decisive_games} "
            f"white_wins={metrics.get('white_wins', 0)} "
            f"black_wins={metrics.get('black_wins', 0)} "
            f"draws={metrics.get('draws', 0)} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
        return result_agents, decisive_games

    algorithm = genetic_algorithm()
    best_agent, loss, gen_wins, _ = algorithm.execute(
        fitness=training_fitness,
        model=model,
        prev_agents=None,
        pop_size=CONFIG["population"],
        generations=CONFIG["generations"],
        mcst_epochs=CONFIG["mcst_epochs"],
        mcst_depth=CONFIG["mcst_depth"],
        benchmark_every_generation=False,
        verbose=False,
        variation_mode=CONFIG["variation_mode"],
        mutation_gene_rate=CONFIG["mutation_gene_rate"],
        mutation_scale=CONFIG["mutation_scale"],
        preserve_global_best=CONFIG["preserve_global_best"],
    )
    training_seconds = time.perf_counter() - training_started

    weight_shapes = _save_weights(best_agent.neural_network, run_dir / "best_weights.npz")
    print(
        f"training_complete seconds={training_seconds:.1f} "
        f"best_fitness={best_agent.fitness:.6f}",
        flush=True,
    )

    stockfish_started = time.perf_counter()
    stockfish_games = _evaluate_stockfish(best_agent)
    stockfish_seconds = time.perf_counter() - stockfish_started

    report = {
        "run_dir": str(run_dir),
        "config": CONFIG,
        "training_seconds": training_seconds,
        "stockfish_evaluation_seconds": stockfish_seconds,
        "best_fitness": _json_float(best_agent.fitness),
        "loss_progression": [_json_float(value) for value in loss],
        "generation_decisive_games": [int(value) for value in gen_wins],
        "total_decisive_games": int(sum(gen_wins)),
        "generation_metrics": generation_metrics,
        "training_games": int(
            CONFIG["generations"]
            * CONFIG["population"]
            * CONFIG["games_per_agent"]
            / 2
        ),
        "weight_shapes": weight_shapes,
        "stockfish_games": stockfish_games,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary={run_dir / 'summary.json'}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
