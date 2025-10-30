class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j] will be the number of ways to get sum j using the first i elements of nums
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1  # There's one way to make sum 0 with 0 elements: use the empty subsequence
        
        for i in range(1, n + 1):
            num = nums[i - 1]
            for j in range(k + 1):
                dp[i][j] = dp[i - 1][j]  # not taking the current number
                if j >= num:
                    dp[i][j] += dp[i - 1][j - num]
                dp[i][j] %= MOD
        
        # Now, dp[n][k] is the number of ways to form sum k using any elements from nums
        # We need to calculate the sum of powers of all subsequences
        total_power = 0
        
        # We use a bitmask to consider all subsequences
        for mask in range(1, 1 << n):
            subsequence_sum = 0
            for i in range(n):
                if mask & (1 << i):
                    subsequence_sum += nums[i]
            if subsequence_sum == k:
                total_power += 1
        
        return total_power % MOD