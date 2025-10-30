from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        
        # total of initial values and total of per-second increments
        total_a = sum(nums1)
        total_b = sum(nums2)
        
        # pair (a, b) and sort by b so that earlier positions get smaller b
        pairs = sorted(zip(nums1, nums2), key=lambda p: p[1])   # ascending by b
        
        # dp[k] = maximum total "benefit" we can obtain using exactly k resets
        # benefit of resetting index i at position k (1-based) in this order is a + k*b
        NEG = -10**18
        dp = [NEG]*(n+1)
        dp[0] = 0
        
        for a, b in pairs:
            # traverse k backwards to avoid overwriting states we still need in this round
            for k in range(n-1, -1, -1):
                if dp[k] == NEG:
                    continue
                cand = dp[k] + a + (k+1)*b
                if cand > dp[k+1]:
                    dp[k+1] = cand
        
        # try all possible times (0 … n); if not feasible by n it will never be
        for t in range(n+1):
            if dp[t] == NEG:
                continue
            # final total = original total + t*total_b - obtained_benefit
            final_sum = total_a + t*total_b - dp[t]
            if final_sum <= x:
                return t
        
        return -1