class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        from collections import Counter
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # We need to consider all unique numbers in nums
        unique_nums = sorted(freq.keys())
        
        # dp[i] will store the sum of all good subsequences ending at unique_nums[i]
        dp = [0] * len(unique_nums)
        
        # Initialize dp with the values of unique_nums because each number itself is a good subsequence
        for i in range(len(unique_nums)):
            dp[i] = unique_nums[i] * freq[unique_nums[i]]
        
        # Calculate the sum of all good subsequences
        for i in range(len(unique_nums)):
            for j in range(i):
                if abs(unique_nums[i] - unique_nums[j]) == 1:
                    dp[i] = (dp[i] + dp[j] * freq[unique_nums[i]]) % MOD
        
        # The result is the sum of all dp values
        result = sum(dp) % MOD
        
        return result