from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        # P will be the alternating prefix sum: P[i] = sum_{k=0..i-1} nums[k] * (-1)^k
        P = 0
        # dp[0] = 0
        # best_even = max(dp[j] - P[j]) over even j seen so far
        # best_odd  = max(dp[j] + P[j]) over odd  j seen so far
        best_even = 0
        best_odd = -10**30
        
        dp_i = 0
        # sign = (-1)^(i-1) for the term nums[i-1] in P update
        sign = 1  # for i=1, (i-1)=0, (-1)^0 = 1
        
        for i, x in enumerate(nums, start=1):
            # update alternating prefix sum P up to i
            P += x * sign
            # dp[i] = max over previous splits ending at j < i
            #      = max(best_even + P, best_odd - P)
            dp_i = max(best_even + P, best_odd - P)
            
            # Now incorporate dp[i] into best_even or best_odd for future i's
            # j = i; if j is even, update best_even = max(best_even, dp[j] - P[j])
            #            if j is odd,  update best_odd  = max(best_odd,  dp[j] + P[j])
            if i % 2 == 0:
                # j = i is even
                best_even = max(best_even, dp_i - P)
            else:
                # j = i is odd
                best_odd = max(best_odd, dp_i + P)
            
            # flip sign for next index
            sign = -sign
        
        return dp_i