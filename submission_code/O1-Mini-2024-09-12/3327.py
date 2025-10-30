from typing import List
import bisect

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, num in enumerate(nums) if num ==1]
        n_ones = len(ones)
        
        if n_ones + maxChanges < k:
            return -1  # Not possible, though constraints say it's possible
        
        # Prefix sums for ones
        prefix = [0]
        for pos in ones:
            prefix.append(prefix[-1] + pos)
        
        def get_cost(l, r):
            m = (l + r) // 2
            median = ones[m]
            # Cost to move all to median
            left_cost = median * (m - l) - (prefix[m] - prefix[l])
            right_cost = (prefix[r] - prefix[m+1]) - median * (r - m -1)
            return left_cost + right_cost
        
        min_cost = float('inf')
        
        if n_ones >= k:
            for i in range(n_ones - k +1):
                cost = get_cost(i, i + k)
                min_cost = min(min_cost, cost)
        else:
            # All existing ones need to be collected
            # Choose aliceIndex to minimize sum distances
            # Optimal at median
            if n_ones ==0:
                min_cost = 0
            else:
                m = n_ones//2
                median = ones[m]
                cost = get_cost(0, n_ones)
                min_cost = cost
            # Add cost for setting new ones
            needed = k - n_ones
            min_cost += needed * 2  # Set and move from adjacent
        return min_cost