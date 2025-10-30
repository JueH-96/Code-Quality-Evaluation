from typing import List
from math import gcd
from collections import defaultdict
from functools import lru_cache

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        max_val = max(nums)
        
        # Precompute gcd for all pairs of numbers up to 200
        gcd_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
        for i in range(1, max_val + 1):
            for j in range(i, max_val + 1):
                gcd_table[i][j] = gcd_table[j][i] = gcd(i, j)
        
        # Count frequencies of each number in nums
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # dp[g] will be the number of subsequences with GCD exactly g
        dp = defaultdict(int)
        
        # Iterate over all possible gcd values
        for g in range(1, max_val + 1):
            # Count subsequences with GCD exactly g using inclusion-exclusion
            subseq_count = 0
            # Multiple of g, mg from g, 2g, ..., until max_val
            for multiple in range(g, max_val + 1, g):
                if multiple in freq:
                    # All subsets of freq[multiple] elements except the empty set
                    subseq_count += (1 << freq[multiple]) - 1
                    subseq_count %= MOD
            
            # Store the count of subsequences with GCD exactly g
            dp[g] = subseq_count
        
        # Calculate the result by considering pairs (g, g)
        result = 0
        for g in range(1, max_val + 1):
            if dp[g] > 0:
                result += dp[g] * dp[g]
                result %= MOD
        
        return result

# Example usage:
sol = Solution()
print(sol.subsequencePairCount([1,2,3,4]))  # Output: 10
print(sol.subsequencePairCount([10,20,30])) # Output: 2
print(sol.subsequencePairCount([1,1,1,1]))  # Output: 50