class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute binomial coefficients
        C = [[0] * (k + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            C[i][0] = 1
            for j in range(1, min(i + 1, k + 1)):
                C[i][j] = (C[i-1][j-1] + C[i-1][j]) % MOD
        
        total = 0
        for i in range(n):
            # Count subsequences where nums[i] is the minimum
            count_min = 0
            for j in range(min(k, n - i)):
                count_min = (count_min + C[n - i - 1][j]) % MOD
            
            # Count subsequences where nums[i] is the maximum
            count_max = 0
            for j in range(min(k, i + 1)):
                count_max = (count_max + C[i][j]) % MOD
            
            # Add contribution
            total = (total + nums[i] * (count_min + count_max)) % MOD
        
        return total