from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Our claim (which one might prove by a greedy exchange argument)
        # is that the overall cost is:
        #    sum_{i=0}^{n-1} min(nums[i], x * i)
        # However, note that since you must “rotate” at least some times,
        # the optimal strategy is to think of each chocolate's cost as being "relaxed"
        # by at most x*i. This formula may be derived by a duality argument.
        total = 0
        for i in range(n):
            total += min(nums[i], x * i)
        return total