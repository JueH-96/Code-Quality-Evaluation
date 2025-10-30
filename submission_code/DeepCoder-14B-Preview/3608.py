import math
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        max_gcd = max(nums)
        # Initialize DP table
        dp = [[0] * (max_gcd + 1) for _ in range(max_gcd + 1)]
        dp[0][0] = 1  # Starting state: both subsequences are empty
        
        for num in nums:
            # Create a new DP table for the current number
            new_dp = [[0] * (max_gcd + 1) for _ in range(max_gcd + 1)]
            for gA in range(max_gcd + 1):
                for gB in range(max_gcd + 1):
                    count = dp[gA][gB]
                    if count == 0:
                        continue
                    # Case 1: do not add the current number to either subsequence
                    new_dp[gA][gB] = (new_dp[gA][gB] + count) % MOD
                    # Case 2: add the current number to the first subsequence (seq1)
                    new_gA = math.gcd(gA, num) if gA != 0 else num
                    new_gB_case2 = gB
                    new_dp[new_gA][new_gB_case2] = (new_dp[new_gA][new_gB_case2] + count) % MOD
                    # Case 3: add the current number to the second subsequence (seq2)
                    new_gB_case3 = math.gcd(gB, num) if gB != 0 else num
                    new_gA_case3 = gA
                    new_dp[new_gA_case3][new_gB_case3] = (new_dp[new_gA_case3][new_gB_case3] + count) % MOD
            dp = new_dp
        
        # Calculate the total number of valid pairs
        total = 0
        for g in range(1, max_gcd + 1):  # g must be non-zero
            total = (total + dp[g][g]) % MOD
        return total