"""Run a paired loss-weight comparison for the native TOP32_v2 repair head."""

import argparse
import json
from pathlib import Path
import sys

from native_layout import DATA, HERE, V5, digest, install, preflight, read_json, relative_path


GROUP = "TOP32_v2"
TIE_HEAD = HERE / "tie_v1" / "matrix_heads.py"
TIE_SHA256 = "dc00b2b4ef433f273d1b807871f9e738e6b104312b0ee60480d6817567d6efab"


def position_config(cfg, loss):
    if cfg["position_weights"] != [1] * 8:
        raise ValueError("original Top-32 position weights differ")
    weights = list(range(8, 0, -1)) if loss == "front" else [1] * 8
    return {**cfg, "position_weights": weights}


def immutable_json(path, value):
    expected = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError(f"sealed configuration differs at {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(expected)


def configure_case(out, loss, freeze):
    native_matrix = V5.parent
    frozen_config = native_matrix / "configs" / f"{GROUP}.json"
    if digest(frozen_config) != freeze["files"][str(frozen_config)]:
        raise ValueError("original Top-32 configuration differs")
    cfg = position_config(read_json(frozen_config), loss)
    runtime_matrix = out / "matrix"
    runtime_config = runtime_matrix / "configs" / f"{GROUP}.json"
    immutable_json(runtime_config, cfg)
    initial = native_matrix / "initial"
    initial_link = runtime_matrix / "initial"
    if not initial_link.is_symlink():
        initial_link.symlink_to(initial, target_is_directory=True)
    if initial_link.resolve() != initial.resolve():
        raise ValueError("initial checkpoint directory differs")
    orders = native_matrix / "epoch_orders.json"
    orders_link = runtime_matrix / "epoch_orders.json"
    if not orders_link.is_symlink():
        orders_link.symlink_to(orders)
    if orders_link.resolve() != orders.resolve():
        raise ValueError("training order differs")
    bound = dict(freeze)
    bound["files"] = {**freeze["files"], str(runtime_config): digest(runtime_config), str(Path(__file__).resolve()): digest(Path(__file__))}
    bound["files"][str(TIE_HEAD.resolve())] = TIE_SHA256
    bound["priority"] = [GROUP]
    bound["top32_loss"] = loss
    bound["top32_selection"] = "one secondary checkpoint selected by equal-token calibration CE"
    immutable_json(out / "runtime_freeze.json", bound)
    return runtime_matrix


def run_case(out, runtime_matrix, phase):
    sys.path[:0] = [str(V5), str(V5.parent), str(DATA / "Spec_v2-main")]
    import common

    # Pin the versioned module before runtime prepends historical search roots.
    sys.path.insert(0, str(TIE_HEAD.parent))
    import matrix_heads
    if Path(matrix_heads.__file__).resolve() != TIE_HEAD.resolve() or digest(TIE_HEAD) != TIE_SHA256:
        raise ValueError("incorrect tie-aware matrix head")

    common.ROOT = out
    common.MATRIX = runtime_matrix
    import priority_worker
    import runtime

    native_loss_for = runtime.loss_for

    def uniform_calibration(head, sample, config, temperature=1.0):
        if temperature is None:
            return native_loss_for(head, sample, {**config, "position_weights": [1] * 8}, temperature)
        return native_loss_for(head, sample, config, temperature)

    runtime.loss_for = uniform_calibration
    selected_path = out / "runs" / GROUP / "train" / "selected.json"
    native_read = priority_worker.read

    def read_secondary(path):
        value = native_read(path)
        if path == selected_path:
            return {"secondary": value["secondary"]}
        return value

    if phase == "eval":
        priority_worker.read = read_secondary
    sys.argv = ["priority_worker.py", "--group", GROUP, "--phase", phase]
    priority_worker.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check-repo", "install", "preflight", "configure", "smoke", "train", "eval"))
    parser.add_argument("--loss", choices=("equal", "front"), default="front")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    if digest(TIE_HEAD) != TIE_SHA256:
        raise ValueError("tie-aware matrix head source differs")
    files = read_json(HERE / "bundled-files.json")
    if args.phase == "install":
        install(files)
        return
    if args.phase == "check-repo":
        for name, expected in files.items():
            if digest(HERE / "native" / relative_path(name)) != expected:
                raise ValueError(f"repository file differs {name}")
        print(f"{len(files)} bundled files passed byte verification")
        return
    freeze = read_json(HERE / "native" / V5.relative_to(DATA) / "runtime_freeze.json")
    preflight(freeze["files"], args.verify_hashes)
    if args.phase == "preflight":
        return
    if args.out is None:
        parser.error("--out is required for configure, smoke, train and eval")
    out = args.out.resolve() / args.loss
    if out.is_relative_to(HERE) or out.is_relative_to(V5):
        raise ValueError("choose an output directory outside the repository and original run")
    matrix = configure_case(out, args.loss, freeze)
    if args.phase == "configure":
        print(out)
        return
    run_case(out, matrix, args.phase)


if __name__ == "__main__":
    main()
