from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7

        # dp[mask][last_index] = number of special partial permutations
        # using elements at indices in mask, ending with nums[last_index].
        # mask is an integer where the i-th bit is set if nums[i] is used.
        # last_index is the index of the last element in the permutation (0 to n-1).
        # Initialize DP table with zeros. The size is 2^n masks by n possible last indices.
        dp = [[0] * n for _ in range(1 << n)]

        # Base case: Permutations of length 1.
        # A single element forms a valid partial permutation of length 1.
        # For each element nums[i] at index i, there is one way to form a permutation
        # consisting only of that element, ending at index i.
        for i in range(n):
            # The mask representing the set containing only the element at index i is 1 << i.
            # The number of such permutations ending with nums[i] is 1.
            dp[1 << i][i] = 1

        # Iterate through masks from 1 up to (1 << n) - 1.
        # Masks represent sets of elements used in a partial permutation.
        # We build permutations by considering masks of increasing size (number of set bits).
        # The loop order of masks from 1 to (1 << n) - 1 naturally ensures that
        # dp values for smaller masks (fewer elements used) are computed before
        # they are needed to compute dp values for larger masks (more elements used).
        for mask in range(1, 1 << n):
            # Iterate through all possible indices (0 to n-1) to identify
            # the index of the last element added to the current partial permutation
            # represented by the current mask.
            for last_index in range(n):
                # Check if the last_index-th bit is set in the current mask.
                # This confirms that nums[last_index] is part of the current
                # partial permutation represented by 'mask'.
                if (mask >> last_index) & 1:
                    # If there is at least one way to form a special partial
                    # permutation using elements in 'mask' that ends with nums[last_index].
                    # We only proceed if this state is reachable.
                    if dp[mask][last_index] > 0:
                        # Try to extend this special partial permutation by adding a new element.
                        # Iterate through all possible indices (0 to n-1) for the next element.
                        for next_index in range(n):
                            # Check if the next_index-th bit is NOT set in the current mask.
                            # This means nums[next_index] has not been used yet in this partial permutation.
                            if not (mask >> next_index) & 1:
                                # Check the divisibility condition between the current last element
                                # nums[last_index] and the potential next element nums[next_index].
                                # A special permutation requires adjacent elements to be divisible by each other.
                                if nums[last_index] % nums[next_index] == 0 or nums[next_index] % nums[last_index] == 0:
                                    # If the condition is met, we can form a new special partial
                                    # permutation by appending nums[next_index] to the current ones.
                                    # The new mask will include the next_index along with the elements in the current mask.
                                    new_mask = mask | (1 << next_index)
                                    # Add the number of ways to reach the current state (mask, last_index)
                                    # to the number of ways to reach the new state (new_mask, next_index).
                                    # We take modulo MOD at each addition to prevent integer overflow.
                                    dp[new_mask][next_index] = (dp[new_mask][next_index] + dp[mask][last_index]) % MOD

        # After filling the DP table, the total number of special permutations
        # of length n is the sum of dp values for the final mask,
        # which includes all elements (mask = (1 << n) - 1), considering any element as the last one.
        total_special_permutations = 0
        # The mask representing that all elements have been used is (1 << n) - 1.
        final_mask = (1 << n) - 1
        # Sum up the counts for all possible last elements in the full permutation.
        for i in range(n):
            # Add the number of special permutations that use all elements
            # and end with nums[i], to the total count.
            total_special_permutations = (total_special_permutations + dp[final_mask][i]) % MOD

        # Return the final count modulo 10^9 + 7.
        return total_special_permutations