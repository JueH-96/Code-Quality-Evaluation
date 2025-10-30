from typing import List
from functools import lru_cache
import sys
sys.setrecursionlimit(10000)

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the rewards so that when we pick one,
        # any candidate from an earlier index would be too small to be picked later.
        rewards = sorted(rewardValues)
        n = len(rewards)
        
        # dp(i, current) returns the maximum final reward we can reach
        # starting with current total reward "current" and using rewards[i:] (each at most once).
        @lru_cache(maxsize=None)
        def dp(i: int, current: int) -> int:
            # If no further candidates are available, return the current total.
            if i == n:
                return current
            # Option 1: Skip the current reward.
            best = dp(i+1, current)
            # Option 2: If the candidate qualifies (i.e. greater than current total), choose it.
            if rewards[i] > current:
                # When we pick rewards[i], new total becomes current + rewards[i].
                best = max(best, dp(i+1, current + rewards[i]))
            return best

        return dp(0, 0)

# ---------------------------------------------------------
# Testing the solution with the provided examples:

if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    rewardValues1 = [1,1,3,3]
    print(sol.maxTotalReward(rewardValues1))  # Expected output: 4

    # Example 2:
    rewardValues2 = [1,6,4,3,2]
    print(sol.maxTotalReward(rewardValues2))  # Expected output: 11