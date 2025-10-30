from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of power of all subsequences of nums.
        The power of an array is the number of its subsequences with sum equal to k.
        """
        MOD = 1_000_000_007
        n = len(nums)

        # dp[j][l]: number of subsequences with sum j and length l
        # j: sum, from 0 to k
        # l: length, from 0 to n
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1  # Base case: empty subsequence has sum 0 and length 0

        # Iterate through each number in nums
        for i, x in enumerate(nums):
            # The maximum possible length of a subsequence using the first i+1 elements is i+1.
            max_len_so_far = i + 1
            
            # We must iterate downwards to use the dp table from the state before processing x
            for j in range(k, x - 1, -1):
                # To form a new subsequence of length l, we add x to an old one of length l-1.
                for l in range(max_len_so_far, 0, -1):
                    dp[j][l] = (dp[j][l] + dp[j - x][l - 1]) % MOD

        # Precompute powers of 2 modulo MOD
        powers_of_2 = [1] * (n + 1)
        for i in range(1, n + 1):
            powers_of_2[i] = (powers_of_2[i - 1] * 2) % MOD

        # Calculate the total sum of powers
        total_power = 0
        # Iterate through all possible lengths l from 1 to n
        for l in range(1, n + 1):
            count = dp[k][l]
            if count > 0:
                # For each subsequence T of nums with sum k and length l, it is a part of 2^(n-l)
                # super-subsequences of nums. Thus, it contributes count * 2^(n-l) to the total sum.
                contribution = powers_of_2[n - l]
                term = (count * contribution) % MOD
                total_power = (total_power + term) % MOD

        return total_power