class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i] will store the number of subsequences with sum i
        dp = [0] * (k + 1)
        dp[0] = 1  # There's one subsequence with sum 0, the empty subsequence
        
        for num in nums:
            for j in range(k, num - 1, -1):
                dp[j] = (dp[j] + dp[j - num]) % MOD
        
        # Calculate the sum of power of all subsequences
        total_power = 0
        for i in range(1, 1 << n):
            subseq_sum = 0
            for j in range(n):
                if i & (1 << j):
                    subseq_sum += nums[j]
            if subseq_sum == k:
                total_power = (total_power + dp[k]) % MOD
        
        return total_power