from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        """
        Compute the difference array D[i] = target[i] - nums[i].
        We treat positive and negative parts separately:
        - For the positive parts P[i] = max(D[i], 0), the minimum number of
          decrement-by-1-on-subarray operations to bring P to all zeros
          is sum of max(P[i] - P[i-1], 0) over i (with P[-1]=0).
        - Similarly for the negative parts N[i] = max(-D[i], 0).
        """
        ans = 0
        pos_prev = 0  # P[i-1]
        neg_prev = 0  # N[i-1]
        
        for a, b in zip(nums, target):
            d = b - a
            # positive part
            pos = d if d > 0 else 0
            # negative part
            neg = -d if d < 0 else 0
            
            # if P[i] > P[i-1], we need (P[i]-P[i-1]) new +1-ops on that segment
            if pos > pos_prev:
                ans += pos - pos_prev
            # if N[i] > N[i-1], we need (N[i]-N[i-1]) new -1-ops on that segment
            if neg > neg_prev:
                ans += neg - neg_prev
            
            pos_prev = pos
            neg_prev = neg
        
        return ans