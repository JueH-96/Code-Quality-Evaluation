from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        freq = Counter(nums)
        max_sum = sum(nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # Base case: one way to make sum 0 (take nothing)
        
        for x in freq:
            k = freq[x]
            m = 1
            while m <= k:
                # Process the virtual item for m copies of x
                for s in range(max_sum, m * x - 1, -1):
                    if s - m * x >= 0:
                        dp[s] = (dp[s] + dp[s - m * x]) % MOD
                k -= m
                m *= 2
        
        # Calculate the sum from l to r inclusive
        total = 0
        for s in range(l, r + 1):
            if s <= max_sum:
                total = (total + dp[s]) % MOD
        return total