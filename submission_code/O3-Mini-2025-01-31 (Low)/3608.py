from math import gcd
from collections import defaultdict
from typing import List

MOD = 10**9 + 7

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll use a DP dictionary keyed by (g1, g2) where:
        # g1: gcd of subsequence1 (or 0 if still empty)
        # g2: gcd of subsequence2 (or 0 if still empty)
        # dp[(g1,g2)] = number of ways to assign the processed indices such that
        # the gcd's are as stated.
        dp = defaultdict(int)
        dp[(0, 0)] = 1
        
        for a in nums:
            new_dp = defaultdict(int)
            # For each state, we have three choices: assign index to neither, subseq1, or subseq2.
            for (g1, g2), count in dp.items():
                # Option 1: assign to neither: state remains (g1, g2)
                new_dp[(g1, g2)] = (new_dp[(g1, g2)] + count) % MOD
                
                # Option 2: assign to subsequence1
                new_g1 = a if g1 == 0 else gcd(g1, a)
                new_dp[(new_g1, g2)] = (new_dp[(new_g1, g2)] + count) % MOD
                
                # Option 3: assign to subsequence2
                new_g2 = a if g2 == 0 else gcd(g2, a)
                new_dp[(g1, new_g2)] = (new_dp[(g1, new_g2)] + count) % MOD
            dp = new_dp
        
        # Count pairs (subseq1, subseq2) with both non-empty (i.e. g1, g2 nonzero)
        # and with equal gcd values.
        ans = 0
        for (g1, g2), count in dp.items():
            if g1 != 0 and g2 != 0 and g1 == g2:
                ans = (ans + count) % MOD
        return ans
       
# For local testing
if __name__ == "__main__":
    sol = Solution()
    print(sol.subsequencePairCount([1,2,3,4]))  # Expected output: 10
    print(sol.subsequencePairCount([10,20,30])) # Expected output: 2
    print(sol.subsequencePairCount([1,1,1,1]))  # Expected output: 50