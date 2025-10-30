from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 1_000_000_007
        
        # From constraints, max cnt_i is 400.
        # This is the maximum number of inversions we need to track.
        MAX_INV = 400
        
        # Use a map for O(1) lookup of requirements.
        req_map = {end: cnt for end, cnt in requirements}
        
        # dp[j] will store the number of permutations of a certain length
        # with j inversions, satisfying all previous requirements.
        
        # Initialize for permutations of length 0.
        # There is one such permutation (the empty one), and it has 0 inversions.
        dp = [0] * (MAX_INV + 1)
        dp[0] = 1
        
        # Iterate from length 1 to n.
        # In each iteration, we compute the dp table for the current `length`
        # based on the dp table for `length - 1`.
        for length in range(1, n + 1):
            # new_dp will store counts for permutations of `length`.
            new_dp = [0] * (MAX_INV + 1)
            
            # The number of permutations of `length` with `j` inversions is the sum
            # of permutations of `length - 1` with `k` inversions, where
            # `j - (length - 1) <= k <= j`.
            # We calculate this using a sliding window sum over the `dp` array.
            current_sum = 0
            for j in range(MAX_INV + 1):
                # Add the next element to the window
                current_sum = (current_sum + dp[j]) % MOD
                
                # Remove the element that falls off the window of size `length`
                if j >= length:
                    current_sum = (current_sum - dp[j - length] + MOD) % MOD
                
                new_dp[j] = current_sum

            # Update dp table for the next iteration
            dp = new_dp
            
            # Apply requirement for the current length, if one exists.
            # The length is `length`, so the end index of the prefix is `length - 1`.
            if length - 1 in req_map:
                required_inversions = req_map[length - 1]
                
                # Filter the dp table: only permutations with the exact required number
                # of inversions are kept. All other counts are set to 0.
                for j in range(MAX_INV + 1):
                    if j != required_inversions:
                        dp[j] = 0
                
                # If required_inversions > MAX_INV, the corresponding dp[j] is already 0,
                # so the whole dp array will become zero, which is correct.
                # This is handled implicitly by our setup.

        # The final answer is the total number of valid permutations of length n.
        # This is the sum of all values in the final dp table.
        # A requirement for end = n-1 is guaranteed, so the final dp table
        # will have at most one non-zero entry.
        total_permutations = sum(dp) % MOD
        
        return total_permutations