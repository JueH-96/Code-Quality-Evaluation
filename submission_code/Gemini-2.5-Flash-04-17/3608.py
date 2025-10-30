import math
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        MAX_VAL = 200 # Max possible value of nums[i], thus max possible GCD > 0
        n = len(nums)

        # dp[g1][g2] stores the number of ways to form disjoint index sets J1, J2
        # from the elements processed so far, such that
        # if J1 is empty, g1 is 0, otherwise gcd({nums[i] for i in J1}) is g1
        # if J2 is empty, g2 is 0, otherwise gcd({nums[i] for i in J2}) is g2
        dp = [[0] * (MAX_VAL + 1) for _ in range(MAX_VAL + 1)]

        # Base case: Before processing any element, both index sets are empty.
        # There is 1 way: select empty set for seq1 indices, empty set for seq2 indices.
        dp[0][0] = 1

        def get_new_gcd(current_gcd, new_element):
            # If current_gcd is 0, it means the subsequence was empty.
            # Adding the new_element (which is >= 1 according to constraints)
            # makes the subsequence [new_element], so the new GCD is new_element.
            if current_gcd == 0:
                return new_element
            # If current_gcd is > 0, it's the GCD of a non-empty subsequence.
            # Adding new_element updates the GCD.
            return math.gcd(current_gcd, new_element)

        for num in nums:
            next_dp = [[0] * (MAX_VAL + 1) for _ in range(MAX_VAL + 1)]
            
            # Iterate through all possible current GCD pairs (g1, g2)
            # States g1 and g2 range from 0 (empty) to MAX_VAL.
            for g1 in range(MAX_VAL + 1):
                for g2 in range(MAX_VAL + 1):
                    if dp[g1][g2] == 0:
                        continue

                    count = dp[g1][g2]

                    # Option 1: Don't include the current number (at the current index) in either subsequence
                    next_dp[g1][g2] = (next_dp[g1][g2] + count) % MOD

                    # Option 2: Include the current number in seq1 (index goes to J1)
                    new_g1 = get_new_gcd(g1, num)
                    # new_g1 will be between 1 and num if g1 was 0, or between 1 and min(g1, num) if g1 > 0.
                    # Since num <= MAX_VAL and g1 <= MAX_VAL, new_g1 will be <= MAX_VAL.
                    next_dp[new_g1][g2] = (next_dp[new_g1][g2] + count) % MOD

                    # Option 3: Include the current number in seq2 (index goes to J2)
                    new_g2 = get_new_gcd(g2, num)
                    # Similar to new_g1, new_g2 will be <= MAX_VAL.
                    next_dp[g1][new_g2] = (next_dp[g1][new_g2] + count) % MOD
            dp = next_dp # Update dp table for the next iteration

        # The total count is the sum of dp[g][g] for g from 1 to MAX_VAL.
        # dp[g][g] for g > 0 counts pairs of disjoint index sets (J1, J2) where:
        # - J1 is non-empty and gcd({nums[i] for i in J1}) = g
        # - J2 is non-empty and gcd({nums[j] for i in J2}) = g
        # This matches the problem requirement: pairs of non-empty disjoint subsequences
        # with equal GCD (which must be > 0).
        total_pairs = 0
        for g in range(1, MAX_VAL + 1):
            total_pairs = (total_pairs + dp[g][g]) % MOD

        return total_pairs