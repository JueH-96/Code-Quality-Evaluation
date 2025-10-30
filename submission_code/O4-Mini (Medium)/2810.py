from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # best[i] will hold the minimal cost to collect type i
        # using up to k shifts. Initialize with k = 0.
        best = nums.copy()
        total = sum(best)
        ans = total  # when k = 0, no shift cost
        
        # Try k = 1 .. n-1 shifts
        for k in range(1, n):
            # For each type i, see if shifting one more time
            # gives a cheaper candidate
            for i in range(n):
                # The candidate cost after one more shift:
                # we pick from index (i - k) mod n in nums
                cand = nums[(i - k) % n]
                if cand < best[i]:
                    # update total sum accordingly
                    total -= (best[i] - cand)
                    best[i] = cand
            # total now is sum of best[i] for this k
            cost_with_k_shifts = total + k * x
            if cost_with_k_shifts < ans:
                ans = cost_with_k_shifts
        
        return ans