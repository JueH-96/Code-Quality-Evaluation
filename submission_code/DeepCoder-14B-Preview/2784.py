class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        if n == 0:
            return 0
        
        # Initialize the DP array S
        S = [0] * n
        
        # Compute the S array
        for j in range(1, n):
            S[j] = (2 * S[j-1] + sorted_nums[j-1]) % MOD
        
        total = 0
        for j in range(n):
            sum_inner = (S[j] + sorted_nums[j]) % MOD
            contribution = (pow(sorted_nums[j], 2, MOD) * sum_inner) % MOD
            total = (total + contribution) % MOD
        
        return total % MOD