class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        
        # Count frequency of each number
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
            
        # Handle zeros separately
        zero_count = freq.get(0, 0)
        if 0 in freq:
            del freq[0]
            
        # Initialize dp array
        max_sum = min(r, sum(k * v for k, v in freq.items()))
        dp = [0] * (max_sum + 1)
        dp[0] = 1
        
        # For each unique number and its frequency
        for num, count in freq.items():
            # For each possible sum from max_sum down to num
            for curr_sum in range(max_sum, num - 1, -1):
                # Add up all possible ways to use 1 to count copies of current number
                for k in range(1, count + 1):
                    if curr_sum >= k * num:
                        dp[curr_sum] = (dp[curr_sum] + dp[curr_sum - k * num]) % MOD
                    else:
                        break
                        
        # Calculate final result
        result = 0
        for i in range(l, min(r + 1, len(dp))):
            result = (result + dp[i]) % MOD
            
        # Multiply by (zero_count + 1) to account for all possible combinations with zeros
        return (result * (zero_count + 1)) % MOD