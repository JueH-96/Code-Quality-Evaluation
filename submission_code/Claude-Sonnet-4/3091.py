class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        
        # Count frequency of each number
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Handle zeros separately - they multiply the number of ways
        zero_count = freq.get(0, 0)
        if 0 in freq:
            del freq[0]
        
        # DP array: dp[i] = number of ways to get sum i
        dp = [0] * (r + 1)
        dp[0] = 1
        
        # For each unique number
        for num, count in freq.items():
            # We need to update dp in reverse order to avoid using updated values
            new_dp = dp[:]
            
            # For each possible sum
            for target_sum in range(num, r + 1):
                # Try adding 1, 2, ..., count copies of current number
                for copies in range(1, min(count, target_sum // num) + 1):
                    prev_sum = target_sum - copies * num
                    new_dp[target_sum] = (new_dp[target_sum] + dp[prev_sum]) % MOD
            
            dp = new_dp
        
        # Sum up all valid sums in range [l, r]
        result = sum(dp[l:r+1]) % MOD
        
        # Multiply by (zero_count + 1) since we can include 0 to zero_count zeros
        result = (result * (zero_count + 1)) % MOD
        
        return result