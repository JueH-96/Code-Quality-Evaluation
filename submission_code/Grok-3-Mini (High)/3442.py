from typing import List
import bisect

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        sorted_rewards = sorted(rewardValues)
        achievable = [0]
        max_reward = 0

        for val in sorted_rewards:
            idx = bisect.bisect_left(achievable, val)
            s_less = achievable[:idx]
            new_sums = [val + s for s in s_less]
            for ns in new_sums:
                if ns > max_reward:
                    max_reward = ns
            achievable.extend(new_sums)
            achievable = sorted(set(achievable))

        return max_reward