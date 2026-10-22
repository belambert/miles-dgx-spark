"""A length penalty on Miles' rule-based reward, for --custom-rm-path length_penalty.reward."""

import copy

from miles.rollout.rm_hub import async_rm

ALPHA = 0.5  # share of the reward a correct answer loses at --rollout-max-response-len


async def reward(args, sample, **kwargs) -> float:
    """--rm-type's reward, scaled down linearly with response length: by ALPHA at the length cap."""
    base_args = copy.copy(args)
    base_args.custom_rm_path = None  # so async_rm falls through to --rm-type instead of back here
    r = await async_rm(base_args, sample, **kwargs)

    # eval samples have no group, and their scores should stay comparable to unpenalized runs
    if sample.group_index is None:
        return r

    # scaling leaves wrong answers at 0: penalizing them by length would reward giving up early
    return r * (1 - ALPHA * sample.response_length / args.rollout_max_response_len)
