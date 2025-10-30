from math import gcd
from typing import List

MOD = 10**9 + 7

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        n = len(nums)
        # Precompute a gcd table for all values 1..200.
        # gcd_table[a][b] will hold gcd(a, b) for a,b in 1..200.
        gcd_table = [[0] * 201 for _ in range(201)]
        for a in range(1, 201):
            for b in range(1, 201):
                gcd_table[a][b] = gcd(a, b)
        
        # We use a DP where each element is processed and we decide for each index whether to:
        #    (a) leave the element unused,
        #    (b) assign it to subsequence 1, or
        #    (c) assign it to subsequence 2.
        # The subsequences are built by selecting indices disjointly.
        # We track for each assignment a pair (g1, g2) where:
        #    • g1 = gcd of all chosen elements for subsequence 1 (0 means subsequence 1 is still empty)
        #    • g2 = gcd of all chosen elements for subsequence 2 (0 means subsequence 2 is still empty)
        #
        # Initially, both subsequences are empty:
        dp = [[0] * 201 for _ in range(201)]
        dp[0][0] = 1

        # Process every number in nums.
        for x in nums:
            new_dp = [[0] * 201 for _ in range(201)]
            for g1 in range(201):
                # Using a local variable for possible speedup.
                for g2 in range(201):
                    ways = dp[g1][g2]
                    if ways:
                        # Option 1: Do not use the current element.
                        new_dp[g1][g2] = (new_dp[g1][g2] + ways) % MOD
                        
                        # Option 2: Add current element x to subsequence 1.
                        # Update gcd for group 1: if group 1 is empty, then new gcd becomes x;
                        # otherwise, update to gcd(g1, x).
                        new_g1 = x if g1 == 0 else gcd_table[g1][x]
                        new_dp[new_g1][g2] = (new_dp[new_g1][g2] + ways) % MOD
                        
                        # Option 3: Add current element x to subsequence 2.
                        new_g2 = x if g2 == 0 else gcd_table[g2][x]
                        new_dp[g1][new_g2] = (new_dp[g1][new_g2] + ways) % MOD
            dp = new_dp

        # Our count should include only those assignments where both subsequences are nonempty,
        # and their gcd’s are equal (and nonzero).
        res = 0
        for d in range(1, 201):
            res = (res + dp[d][d]) % MOD
        return res


# For testing the solution with the provided examples:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    # For nums = [1, 2, 3, 4], the number of valid pairs is 10.
    print(sol.subsequencePairCount([1, 2, 3, 4]))  # Expected output: 10
    
    # Example 2:
    # For nums = [10, 20, 30], the number of valid pairs is 2.
    print(sol.subsequencePairCount([10, 20, 30]))  # Expected output: 2

    # Example 3:
    # For nums = [1, 1, 1, 1], the number of valid pairs is 50.
    print(sol.subsequencePairCount([1, 1, 1, 1]))   # Expected output: 50