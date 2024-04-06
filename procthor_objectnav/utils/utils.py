import os
import sys, pdb
import torch.nn as nn
from allenact.utils.system import get_logger
from allenact.base_abstractions.misc import ActorCriticOutput


class ForkedPdb(pdb.Pdb):  # used for real episode logging
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def debug_model_info(
    model: nn.Module, trainable: bool = True, use_logger: bool = True, **kwargs
):
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return
    debug_msg = (
        f"{model}"
        + (
            f"\nTrainable Parameters: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        * trainable
    )
    if use_logger:
        get_logger().debug("".join([str(t) for t in debug_msg])[:-1], **kwargs)
    else:
        print(debug_msg, **kwargs)


def log_ac_return(ac: ActorCriticOutput, task_id_obs):
    if os.getenv("LOGGING_DIR") is None:
        d = "output"
    else:
        d = os.getenv("LOGGING_DIR")
    os.makedirs(f"{d}/ac-data/", exist_ok=True)

    assert len(task_id_obs.shape) == 3

    for i in range(len(task_id_obs[0])):
        task_id = "".join(
            [
                chr(int(k))
                for k in task_id_obs[0, i]
                if chr(int(k)) != " "
            ]
        )

        with open(f"{d}/ac-data/{task_id}.txt", "a") as f:
            estimated_value = ac.values[0, i].item()
            policy = nn.functional.softmax(
                ac.distributions.logits[0, i]
            ).tolist()
            f.write(",".join(map(str, policy + [estimated_value])) + "\n")
