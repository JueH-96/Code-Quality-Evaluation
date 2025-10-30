import math
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        MAX_VAL = 200 # Max value of nums[i] based on constraints (1 <= nums[i] <= 200)

        # dp[g1][g2] stores the number of ways to form two disjoint subsequences
        # such that:
        # - The first subsequence (seq1) has GCD g1.
        # - The second subsequence (seq2) has GCD g2.
        # A GCD value of 0 signifies an empty subsequence.
        # math.gcd(0, x) returns x, which conveniently handles the first element
        # added to an empty subsequence.
        
        # Initialize dp table.
        # dp[0][0] = 1 represents one way to have two empty subsequences
        # before processing any numbers. All other states are initially 0.
        dp = [[0] * (MAX_VAL + 1) for _ in range(MAX_VAL + 1)]
        dp[0][0] = 1

        # Iterate through each number in the input array nums
        for num in nums:
            # Create a new dp table for the current iteration.
            # This is crucial to avoid using values that were already updated
            # by the current 'num' within the same iteration.
            new_dp = [[0] * (MAX_VAL + 1) for _ in range(MAX_VAL + 1)]

            # Iterate over all possible previous GCD pairs (g1, g2)
            for g1 in range(MAX_VAL + 1):
                for g2 in range(MAX_VAL + 1):
                    # If there are no ways to reach this (g1, g2) state, skip it
                    if dp[g1][g2] == 0:
                        continue 

                    current_ways = dp[g1][g2]

                    # Option 1: 'num' is not used in any subsequence (neither seq1 nor seq2).
                    # The GCDs g1 and g2 remain unchanged.
                    new_dp[g1][g2] = (new_dp[g1][g2] + current_ways) % MOD

                    # Option 2: 'num' is added to seq1.
                    # This implies 'num' cannot be added to seq2 in this specific path.
                    # Calculate the new GCD for seq1.
                    # If g1 was 0 (seq1 was empty), its GCD becomes num. Otherwise, it's gcd(g1, num).
                    new_g1 = math.gcd(g1, num) if g1 != 0 else num
                    new_dp[new_g1][g2] = (new_dp[new_g1][g2] + current_ways) % MOD

                    # Option 3: 'num' is added to seq2.
                    # This implies 'num' cannot be added to seq1 in this specific path.
                    # Calculate the new GCD for seq2.
                    # If g2 was 0 (seq2 was empty), its GCD becomes num. Otherwise, it's gcd(g2, num).
                    new_g2 = math.gcd(g2, num) if g2 != 0 else num
                    new_dp[g1][new_g2] = (new_dp[g1][new_g2] + current_ways) % MOD
            
            # After processing all choices for the current 'num', update 'dp' for the next iteration.
            dp = new_dp 

        # The final answer is the sum of dp[g][g] for all possible GCD values 'g'.
        # We are looking for pairs where both seq1 and seq2 are non-empty (hence g > 0)
        # and their GCDs are equal.
        total_pairs = 0
        for g in range(1, MAX_VAL + 1):
            total_pairs = (total_pairs + dp[g][g]) % MOD

        return total_pairs