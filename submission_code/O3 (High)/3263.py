from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        """
        We have to split the whole array into three consecutive non-empty parts.
        Let the split points be 0 … i-1 | i … j-1 | j … n-1
        (so indices i and j are the first elements of the 2nd and 3rd sub-arrays).

        The total cost is
            cost = nums[0] + nums[i] + nums[j]

        nums[0] is fixed.  
        Therefore we only need the pair (i, j) with 1 ≤ i < j ≤ n-1 that minimises
            nums[i] + nums[j].

        With n ≤ 50 an 𝑂(n²) search is more than fast enough, but we can do it in
        𝑂(n) by remembering the smallest value seen so far while we scan for j.
        """
        n = len(nums)

        # smallest value among nums[1 .. j-1] while we sweep j from 2 .. n-1
        best_prefix = nums[1]
        best_pair_sum = float('inf')

        for j in range(2, n):
            # cost of choosing the previously best i together with this j
            best_pair_sum = min(best_pair_sum, best_prefix + nums[j])
            # update the best i for future j's
            best_prefix = min(best_prefix, nums[j])

        return nums[0] + best_pair_sum