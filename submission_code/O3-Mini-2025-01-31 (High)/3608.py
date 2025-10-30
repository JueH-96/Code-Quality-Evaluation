from typing import List
import math
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        max_val = 201  # because nums[i] and gcd values go up to 200
        
        # Precompute gcd for all pairs (a,b) with 0 <= a, b <= 200.
        # Note: by convention, math.gcd(0, x) == x.
        gcd_table = [[0] * max_val for _ in range(max_val)]
        for i in range(max_val):
            for j in range(max_val):
                gcd_table[i][j] = math.gcd(i, j)
        
        # We'll use dynamic programming where the state (g1, g2)
        # represents the gcd of the elements chosen in group1 and group2 so far.
        # The initial state is (0,0) meaning neither subsequence has any chosen elements.
        dp = {(0, 0): 1}
        
        # For each number, we decide how to use it:
        #   1. Do not use it.
        #   2. Append it to subsequence 1 (update gcd: new_g1 = gcd(old_g1, x)).
        #   3. Append it to subsequence 2 (update gcd: new_g2 = gcd(old_g2, x)).
        # Note that by definition gcd(0, x) = x so our update also works for an empty subsequence.
        for x in nums:
            new_dp = defaultdict(int)
            for (g1, g2), count in dp.items():
                # Option 1: Do not use x
                new_dp[(g1, g2)] = (new_dp[(g1, g2)] + count) % mod
                # Option 2: Use x in subsequence1
                new_g1 = gcd_table[g1][x]
                new_dp[(new_g1, g2)] = (new_dp[(new_g1, g2)] + count) % mod
                # Option 3: Use x in subsequence2
                new_g2 = gcd_table[g2][x]
                new_dp[(g1, new_g2)] = (new_dp[(g1, new_g2)] + count) % mod
            dp = new_dp
        
        # We must count only those assignments in which both subsequences are non-empty
        # and have equal gcd. Our state (g1, g2) tells us the current gcd of each subsequence,
        # and a subsequence is non-empty if its gcd is nonzero.
        ans = 0
        for (g1, g2), count in dp.items():
            if g1 != 0 and g2 != 0 and g1 == g2:
                ans = (ans + count) % mod
        return ans