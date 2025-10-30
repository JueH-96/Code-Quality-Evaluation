import collections
from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7

        # dp[mask][i] stores the number of special permutations using elements
        # represented by 'mask', with 'nums[i]' being the last element.
        # The mask ranges from 0 to (1 << n) - 1.
        # The index i ranges from 0 to n - 1.
        dp = [[0] * n for _ in range(1 << n)]

        # Base cases: A single element forms a valid special permutation of length 1.
        # Initialize dp states for permutations of length 1.
        for i in range(n):
            # (1 << i) creates a mask with only the i-th bit set.
            # This represents using only nums[i].
            dp[1 << i][i] = 1

        # Iterate through masks. Masks are processed in increasing order,
        # which means shorter permutations are processed before longer ones.
        for mask in range(1, 1 << n):
            for i in range(n):
                # If nums[i] is not included in the current mask, or if there's no way
                # to form a permutation ending with nums[i] using this mask, skip.
                # (mask & (1 << i)) checks if the i-th bit is set in the mask.
                if not (mask & (1 << i)) or dp[mask][i] == 0:
                    continue

                # Try to append a new element nums[j] to the permutation ending with nums[i].
                for j in range(n):
                    # nums[j] must not already be in the current mask.
                    if not (mask & (1 << j)):
                        # Check the divisibility condition between nums[i] and nums[j].
                        if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                            # Calculate the new mask by adding nums[j].
                            new_mask = mask | (1 << j)
                            # Add the count of permutations ending with nums[i] to
                            # the count of permutations ending with nums[j] with the new mask.
                            dp[new_mask][j] = (dp[new_mask][j] + dp[mask][i]) % MOD
        
        # The total number of special permutations is the sum of all dp states
        # where all elements have been used (mask = (1 << n) - 1).
        total_special_permutations = 0
        full_mask = (1 << n) - 1 # Mask where all n bits are set
        for i in range(n):
            total_special_permutations = (total_special_permutations + dp[full_mask][i]) % MOD

        return total_special_permutations