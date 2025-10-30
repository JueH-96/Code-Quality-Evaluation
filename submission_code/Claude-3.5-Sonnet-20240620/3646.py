class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        
        # dp[i][j] represents the sum of good subsequences ending with number i
        # and having length j
        dp = [[0] * (len(nums) + 1) for _ in range(max_num + 1)]
        
        # Count occurrences of each number
        count = [0] * (max_num + 1)
        for num in nums:
            count[num] += 1
        
        total_sum = 0
        
        # Initialize base cases: subsequences of length 1
        for i in range(max_num + 1):
            if count[i] > 0:
                dp[i][1] = count[i]
                total_sum = (total_sum + count[i]) % MOD
        
        # Fill the dp table
        for length in range(2, len(nums) + 1):
            for num in range(max_num + 1):
                if count[num] > 0:
                    # Add subsequences ending with 'num'
                    dp[num][length] = count[num] * (
                        (dp[num-1][length-1] if num > 0 else 0) +
                        (dp[num+1][length-1] if num < max_num else 0)
                    ) % MOD
                    
                    total_sum = (total_sum + dp[num][length]) % MOD
        
        return total_sum