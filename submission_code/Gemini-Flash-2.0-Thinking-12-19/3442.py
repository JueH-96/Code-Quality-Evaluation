from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        memo = {}

        def solve(current_reward, marked_indices_tuple):
            marked_indices = set(marked_indices_tuple)
            state = (current_reward, marked_indices_tuple)
            if state in memo:
                return memo[state]

            max_r = current_reward
            for i in range(n):
                if i not in marked_indices and rewardValues[i] > current_reward:
                    next_marked_indices = set(marked_indices)
                    next_marked_indices.add(i)
                    reward = solve(current_reward + rewardValues[i], tuple(sorted(list(next_marked_indices))))
                    max_r = max(max_r, reward)
            
            memo[state] = max_r
            return max_r

        return solve(0, tuple())