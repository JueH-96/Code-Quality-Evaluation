import bisect
from typing import List

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        
        # Collect all unique differences between pairs
        differences = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = nums_sorted[j] - nums_sorted[i]
                differences.add(d)
        
        total = 0
        for d in differences:
            f_d = self.compute_f(d, nums_sorted, k)
            f_d_plus = self.compute_f(d + 1, nums_sorted, k)
            contribution = d * (f_d - f_d_plus)
            total += contribution
        
        return total % MOD
    
    def compute_f(self, D, A, k):
        n = len(A)
        if k == 0:
            return 0
        dp = [[0] * (k + 1) for _ in range(n)]
        for i in range(n):
            dp[i][1] = 1
        
        for j in range(2, k + 1):
            # Compute prefix sums for the previous layer (j-1)
            prefix = [0] * n
            prefix[0] = dp[0][j-1]
            for m in range(1, n):
                prefix[m] = prefix[m-1] + dp[m][j-1]
            
            for i in range(n):
                target = A[i] - D
                # Find the largest m < i where A[m] <= target
                idx = bisect.bisect_right(A, target, 0, i) - 1
                if idx < 0:
                    total = 0
                else:
                    total = prefix[idx]
                dp[i][j] = total
        
        res = 0
        for i in range(n):
            res += dp[i][k]
        return res