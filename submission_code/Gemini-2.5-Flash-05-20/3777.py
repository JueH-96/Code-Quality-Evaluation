from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        MAX_N = len(nums) 
        MAX_VAL_ABS = 12 # nums[i] is between 0 and 12

        # The maximum possible alternating sum for a subsequence is MAX_N * MAX_VAL_ABS.
        # The minimum possible alternating sum is MAX_N * (-MAX_VAL_ABS).
        # We need an offset to map these sums to non-negative indices for the DP table.
        SUM_OFFSET = MAX_N * MAX_VAL_ABS
        
        # The sum dimension for our DP table will range from 0 to 2 * SUM_OFFSET.
        MAX_SUM_DP_DIM = 2 * SUM_OFFSET + 1

        # dp[current_alt_sum_shifted][length_parity] stores the maximum product.
        # length_parity: 0 for even length, 1 for odd length.
        # Initialize with None to indicate unreachable states (no valid product found yet).
        dp = [[None] * 2 for _ in range(MAX_SUM_DP_DIM)]

        # Base case: conceptual empty subsequence.
        # Sum is 0 (shifted to SUM_OFFSET), length is 0 (even, parity 0), product is 1 (multiplicative identity).
        # This state is essential to correctly calculate products for subsequences starting with the first element.
        dp[SUM_OFFSET][0] = 1

        # Iterate through each number in the input array.
        for num in nums:
            # Create a new DP table for the current iteration.
            # This allows us to correctly handle choices for 'num' without affecting states
            # that 'num' itself might depend on in the current iteration.
            new_dp = [[None] * 2 for _ in range(MAX_SUM_DP_DIM)]

            # First, carry over states from the previous DP table.
            # This covers subsequences that *do not* include the current 'num'.
            for s_idx in range(MAX_SUM_DP_DIM):
                for p_idx in range(2):
                    if dp[s_idx][p_idx] is not None:
                        # Copy the product, taking the maximum if this state was already reached
                        # (e.g., from an earlier path not including the previous 'num').
                        if new_dp[s_idx][p_idx] is None:
                            new_dp[s_idx][p_idx] = dp[s_idx][p_idx]
                        else:
                            new_dp[s_idx][p_idx] = max(new_dp[s_idx][p_idx], dp[s_idx][p_idx])

            # Now, iterate through the previous DP states and consider including the current 'num'.
            # We must iterate over 'dp' (the previous state) to ensure valid transitions.
            for s_idx in range(MAX_SUM_DP_DIM):
                for p_idx in range(2): # p_idx is the length_parity of the subsequence *before* adding 'num'
                    current_product = dp[s_idx][p_idx]
                    
                    if current_product is None:
                        continue # This state is unreachable with a valid product

                    # Calculate the new product if 'num' is included.
                    new_product = current_product * num

                    # Pruning: If the new product exceeds the given limit, this path is invalid.
                    # A product of 0 (if 'num' is 0 or 'current_product' was 0) is always <= limit (since limit >= 1).
                    if new_product > limit:
                        continue

                    # Determine how 'num' affects the alternating sum.
                    # If p_idx is 0 (current subsequence has even length), 'num' is added at an even index (0, 2, ...), so it contributes positively.
                    # If p_idx is 1 (current subsequence has odd length), 'num' is added at an odd index (1, 3, ...), so it contributes negatively.
                    sum_change = num if p_idx == 0 else -num
                    new_sum_shifted = s_idx + sum_change
                    
                    # The new subsequence's length parity will be the opposite of the current one.
                    new_parity = (p_idx + 1) % 2 

                    # Ensure the new shifted sum index is within the valid bounds of our DP table.
                    if 0 <= new_sum_shifted < MAX_SUM_DP_DIM:
                        # Update new_dp with the maximum product found for this new state.
                        if new_dp[new_sum_shifted][new_parity] is None or new_product > new_dp[new_sum_shifted][new_parity]:
                            new_dp[new_sum_shifted][new_parity] = new_product
            
            # After processing all states for the current 'num', update 'dp' to 'new_dp'
            # for the next iteration (processing the next number in nums).
            dp = new_dp

        # After processing all numbers, retrieve the maximum product for the target sum 'k'.
        target_sum_idx = k + SUM_OFFSET
        
        # Check if the target sum 'k' is within the range of possible sums that our DP table covers.
        if not (0 <= target_sum_idx < MAX_SUM_DP_DIM):
            return -1 # Target k is outside the possible sum range for any subsequence.

        max_valid_product = -1 # Initialize with -1 as per problem specification (no solution found).

        # Check for maximum product from both even and odd length subsequences for the target sum 'k'.
        
        # Check even length subsequences (length_parity 0)
        p0 = dp[target_sum_idx][0]
        if p0 is not None:
            # The problem requires a non-empty subsequence.
            # The base case `dp[SUM_OFFSET][0] = 1` represents an empty subsequence (sum 0, product 1).
            # We must ensure that if `p0` is 1 and `k` is 0, this does not represent the empty subsequence.
            # The condition `p0 != 1 or k != 0` handles this:
            # - If `p0` is not 1, it's a product from a non-empty subsequence.
            # - If `p0` is 1 but `k` is not 0, it means a non-empty subsequence (e.g., `[1]` for `k=1`) has product 1, which is valid.
            # - If `p0` is 1 AND `k` is 0, the condition `(p0 != 1 or k != 0)` evaluates to `(False or False)`, which is `False`.
            #   In this case, `max_valid_product` is not updated by this `p0`, correctly excluding the empty subsequence.
            if p0 != 1 or k != 0:
                max_valid_product = max(max_valid_product, p0)
        
        # Check odd length subsequences (length_parity 1)
        p1 = dp[target_sum_idx][1]
        if p1 is not None:
            # Apply the same logic as for p0 to ensure we only consider non-empty subsequences.
            if p1 != 1 or k != 0:
                max_valid_product = max(max_valid_product, p1)

        return max_valid_product