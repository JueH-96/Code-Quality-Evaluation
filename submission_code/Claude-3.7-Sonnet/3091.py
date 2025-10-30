class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        
        # Count frequencies of each number
        from collections import Counter
        freq = Counter(nums)
        
        # Handle zeros separately since they don't change the sum
        zero_count = freq.pop(0, 0)
        
        # Initialize dp array where dp[i] = number of ways to make sum i
        dp = [1] + [0] * r
        
        # For each unique number and its frequency
        for num, count in freq.items():
            # For each remainder when divided by num
            for start in range(min(num, r + 1)):
                # Use sliding window to calculate sums efficiently
                window_sum = 0
                
                # First calculate for the first "count+1" positions
                for j in range((min(r + 1, start + (count + 1) * num) - start) // num):
                    idx = start + j * num
                    window_sum = (window_sum + dp[idx]) % MOD
                    dp[idx] = window_sum
                
                # Then use sliding window for the rest
                for j in range((min(r + 1, start + (count + 1) * num) - start) // num, (r + 1 - start) // num):
                    idx = start + j * num
                    idx_to_remove = idx - (count + 1) * num
                    window_sum = (window_sum - dp[idx_to_remove] + dp[idx]) % MOD
                    dp[idx] = window_sum
        
        # Account for zeros - each 0 can be either included or excluded
        # so we multiply by (zero_count + 1)
        zero_multiplier = zero_count + 1
        
        # Sum up the counts for all sums from l to r
        result = 0
        for i in range(l, min(r + 1, len(dp))):
            result = (result + dp[i]) % MOD
        
        return (result * zero_multiplier) % MOD