from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # Initialize dp and aux tables with negative infinity
        dp = [ [ -float('inf') ] * (k + 1) for _ in range(n + 1) ]
        aux = [ [ -float('inf') ] * (k + 1) for _ in range(n + 1) ]
        dp[0][0] = 0  # Base case: 0 subarrays sum to 0 at position 0
        
        # max_prev_val[k] tracks the maximum value of (dp[s][k-1] - prefix[s]) for s <= i - m
        max_prev_val = [ -float('inf') ] * (k + 1)
        
        for i in range(1, n + 1):
            # Update max_prev_val for each k based on current i and s = i - m
            if i >= m:
                s = i - m
                # Iterate through all possible k values up to given k
                for kk in range(1, k + 1):
                    prev_kk = kk - 1
                    if prev_kk >= 0:
                        current_candidate = dp[s][prev_kk] - prefix[s]
                        if current_candidate > max_prev_val[kk]:
                            max_prev_val[kk] = current_candidate
            
            # Process transitions for each k
            for kk in range(1, k + 1):
                # Check if it's possible to have kk subarrays each of length >= m
                if kk * m > i:
                    dp[i][kk] = -float('inf')
                    continue
                
                # Compute options for aux[i][kk]
                option1 = aux[i-1][kk] + nums[i-1] if aux[i-1][kk] != -float('inf') else -float('inf')
                if max_prev_val[kk] != -float('inf'):
                    option2 = max_prev_val[kk] + prefix[i]
                else:
                    option2 = -float('inf')
                
                aux[i][kk] = max(option1, option2)
                # Update dp[i][kk]
                dp[i][kk] = max(dp[i-1][kk], aux[i][kk])
        
        return dp[n][k]