from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        def count_subsequences_with_sum(nums, target):
            n = len(nums)
            dp = [0] * (target + 1)
            dp[0] = 1  # There's one way to make sum 0, by choosing no elements
            
            for num in nums:
                for j in range(target, num - 1, -1):
                    dp[j] = (dp[j] + dp[j - num]) % MOD
            
            return dp[target]
        
        total_power = 0
        n = len(nums)
        
        # Generate all subsequences
        for i in range(1, 1 << n):
            subseq = []
            for j in range(n):
                if i & (1 << j):
                    subseq.append(nums[j])
            total_power = (total_power + count_subsequences_with_sum(subseq, k)) % MOD
        
        return total_power