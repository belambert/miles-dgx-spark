"""DAPO's overlong penalty on top of Miles' rule-based reward, for --custom-rm-path length_penalty.reward."""

import copy

from miles.rollout.rm_hub import async_rm

BUFFER = 0.25  # share of --rollout-max-response-len over which the penalty ramps from 0 to -1


async def reward(args, sample, **kwargs) -> float:
    """--rm-type's reward, minus a penalty growing linearly to 1 over the last BUFFER of the length cap."""
    base_args = copy.copy(args)
    base_args.custom_rm_path = None  # so async_rm falls through to --rm-type instead of back here
    r = await async_rm(base_args, sample, **kwargs)

    # eval samples have no group, and their scores should stay comparable to unpenalized runs
    if sample.group_index is None:
        return r

    cap = args.rollout_max_response_len
    start = cap * (1 - BUFFER)
    return r - min(max(sample.response_length - start, 0) / (cap - start), 1)
