class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        
        # Count the frequency of each number
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # Handle zero separately
        zero_count = freq.get(0, 0)
        if 0 in freq:
            del freq[0]
        
        # Initialize dp array
        max_sum = min(sum(nums), r)
        dp = [0] * (max_sum + 1)
        dp[0] = 1
        
        # Dynamic programming
        for num, count in freq.items():
            for i in range(max_sum, num - 1, -1):
                for j in range(1, count + 1):
                    if i >= j * num:
                        dp[i] = (dp[i] + dp[i - j * num]) % MOD
                    else:
                        break
        
        # Calculate the final result
        result = sum(dp[l:r+1]) % MOD
        
        # Multiply by (zero_count + 1) to account for subsets including zeros
        result = (result * (zero_count + 1)) % MOD
        
        return result