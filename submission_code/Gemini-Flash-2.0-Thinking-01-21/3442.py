from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        memo = {}

        def get_max_reward(current_reward, unmarked_indices_tuple):
            unmarked_indices = set(unmarked_indices_tuple)
            state = (current_reward, unmarked_indices_tuple)
            if state in memo:
                return memo[state]

            max_r = current_reward
            possible_to_pick = False
            for index in list(unmarked_indices):
                if rewardValues[index] > current_reward:
                    possible_to_pick = True
                    next_unmarked_indices = set(unmarked_indices)
                    next_unmarked_indices.remove(index)
                    reward_if_picked = get_max_reward(current_reward + rewardValues[index], tuple(sorted(list(next_unmarked_indices))))
                    max_r = max(max_r, reward_if_picked)

            memo[state] = max_r
            return max_r

        initial_unmarked_indices = tuple(range(n))
        return get_max_reward(0, initial_unmarked_indices)