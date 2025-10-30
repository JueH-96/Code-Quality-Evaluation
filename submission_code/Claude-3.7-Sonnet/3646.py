class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][v] = sum of elements in good subsequences ending at index i with value v
        dp = [{} for _ in range(n)]
        # count[i][v] = number of good subsequences ending at index i with value v
        count = [{} for _ in range(n)]
        
        # Process each element
        for i in range(n):
            v = nums[i]
            
            # Initialize the singleton subsequence
            dp[i][v] = v
            count[i][v] = 1
            
            # Extend good subsequences ending at previous positions
            for j in range(i):
                for prev_val in dp[j]:
                    if abs(prev_val - v) == 1:
                        # Get the sum and count of subsequences ending at j with value prev_val
                        old_sum = dp[j][prev_val]
                        old_count = count[j][prev_val]
                        
                        # Calculate the contribution when we extend these subsequences
                        extended_sum = old_sum + v * old_count
                        
                        dp[i][v] = (dp[i][v] + extended_sum) % MOD
                        count[i][v] = (count[i][v] + old_count) % MOD
        
        # Sum up all good subsequences
        total_sum = 0
        for i in range(n):
            for v in dp[i]:
                total_sum = (total_sum + dp[i][v]) % MOD
        
        return total_sum