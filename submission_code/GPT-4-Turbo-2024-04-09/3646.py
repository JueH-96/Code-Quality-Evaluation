class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        from collections import defaultdict
        
        # Dictionary to count occurrences of each number
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
        
        # Dictionary to store the sum of all good subsequences ending with each number
        dp = defaultdict(int)
        
        # Initialize dp with single element subsequences
        for num in count:
            dp[num] = num * count[num] % MOD
        
        # Process numbers in sorted order to ensure we only form valid subsequences
        sorted_nums = sorted(count.keys())
        
        # Calculate the sum of good subsequences for each number
        for num in sorted_nums:
            # Good subsequences can extend to num+1 if num+1 exists
            if num + 1 in count:
                dp[num + 1] = (dp[num + 1] + dp[num]) % MOD
        
        # Sum all dp values to get the result
        result = sum(dp.values()) % MOD
        return result