from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        """
        Calculates the number of special permutations of a given list of distinct positive integers.
        A permutation is special if for every adjacent pair of elements (a, b),
        either a % b == 0 or b % a == 0.
        """
        n = len(nums)
        MOD = 10**9 + 7

        # memo stores the results of our dp function to avoid recomputation.
        # The state is a tuple (mask, last_idx).
        memo = {}

        def dp(mask: int, last_idx: int) -> int:
            """
            Recursive function to count special permutations.
            :param mask: A bitmask representing the set of used numbers' indices.
            :param last_idx: The index of the last number added to the permutation.
            :return: The number of ways to complete a special permutation from this state.
            """
            # Base case: if all numbers are used, we have formed one valid special permutation.
            if mask == (1 << n) - 1:
                return 1
            
            # If this state has been computed, return the cached value.
            state = (mask, last_idx)
            if state in memo:
                return memo[state]

            count = 0
            # Iterate through all numbers to find the next valid number for the permutation.
            for next_idx in range(n):
                # Check if the number at next_idx has not been used yet.
                if not (mask & (1 << next_idx)):
                    # Check the special condition with the last number added.
                    if nums[next_idx] % nums[last_idx] == 0 or nums[last_idx] % nums[next_idx] == 0:
                        # If valid, recurse for the next state and add the result to the count.
                        new_mask = mask | (1 << next_idx)
                        count = (count + dp(new_mask, next_idx)) % MOD
            
            # Cache the result for the current state and return it.
            memo[state] = count
            return count

        # The total count is the sum of special permutations starting with each possible number.
        total_count = 0
        for i in range(n):
            # Start a permutation with nums[i].
            # The initial mask has the i-th bit set.
            # The last element added is at index i.
            total_count = (total_count + dp(1 << i, i)) % MOD

        return total_count