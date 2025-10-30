import math
from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        n = len(nums)
        
        # Calculate the maximum possible absolute alternating sum for a subsequence of length up to n.
        # For a subsequence of length L, the max absolute alternating sum is ceil(L/2) * max(nums[i]).
        # Since nums[i] >= 0, the maximum positive contribution from an element is max(nums[i]) and the maximum negative contribution is 0.
        # The worst-case (maximum absolute) sum is achieved by picking the largest allowed value (12) at positions that maximize its contribution (even indices for positive sum, odd indices for negative sum).
        # For a subsequence of length L, max positive sum is ceil(L/2) * max(nums[i]), min negative sum is -ceil(L/2) * max(nums[i]).
        # The maximum possible absolute sum across all subsequence lengths <= n is bounded by ceil(n/2) * 12.
        # Note: Using 12 as the upper bound for nums[i] based on problem constraints.
        max_possible_subsequence_sum_abs = math.ceil(n / 2) * 12 if n > 0 else 0
        
        # We use an offset to handle negative sums in the DP table index.
        # The sum range is [-max_possible_subsequence_sum_abs, max_possible_subsequence_sum_abs].
        # The offsetted sum range is [0, 2 * max_possible_subsequence_sum_abs].
        OFFSET = max_possible_subsequence_sum_abs
        SUM_RANGE_SIZE = 2 * max_possible_subsequence_sum_abs + 1
        
        # dp[length][s_offset] stores the maximum product of a subsequence with length `length`
        # and alternating sum `s_offset - OFFSET`, using elements considered so far.
        # We use space optimization, keeping only the previous (dp_prev) and current (dp_curr) states.
        # Initialize dp_prev with -1 to indicate an unreachable state or an invalid product <= limit.
        # The dimensions are (n + 1) for length (0 to n) and SUM_RANGE_SIZE for sum.
        dp_prev = [[-1] * SUM_RANGE_SIZE for _ in range(n + 1)]
        
        # Base case: empty subsequence (length 0, sum 0, product 1).
        # Product 1 is the multiplicative identity, necessary for correctly calculating products
        # when adding the first element. The problem asks for non-empty subsequences
        # in the final result, which we check for when collecting the answer.
        dp_prev[0][OFFSET] = 1
        
        # Iterate through numbers in nums using index `i`
        for i in range(n):
            num = nums[i]
            
            # Create a new DP table `dp_curr` for states including the current number `nums[i]`.
            # This table will store maximum products using elements from `nums[0...i]`.
            dp_curr = [[-1] * SUM_RANGE_SIZE for _ in range(n + 1)]
            
            # Option 1: Don't include nums[i].
            # States reachable using nums[0...i] without using nums[i] are exactly
            # the states reachable using nums[0...i-1].
            # The lengths possible using nums[0...i-1] range from 0 up to `i`.
            for length in range(i + 1): 
                for s_offset in range(SUM_RANGE_SIZE):
                    # Copy the state from the previous DP table.
                    dp_curr[length][s_offset] = dp_prev[length][s_offset]

            # Option 2: Include nums[i].
            # We form a new subsequence by adding nums[i] to a valid subsequence
            # that used elements from nums[0...i-1}.
            # Let the previous valid subsequence have length `prev_length` and sum `prev_sum`.
            # `prev_length` can range from 0 up to `i`.
            # The new subsequence length will be `new_length = prev_length + 1`.
            
            # Iterate through all possible previous lengths (from 0 up to the max possible length using nums[0...i-1])
            for prev_length in range(i + 1): 
                # Iterate through all possible previous sum offsets
                for prev_s_offset in range(SUM_RANGE_SIZE):
                    prev_prod = dp_prev[prev_length][prev_s_offset]
                    
                    # Only consider reachable previous states (those with a valid product >= 0)
                    if prev_prod != -1: 
                        new_length = prev_length + 1
                        
                        # The new length must be within the maximum possible length (n)
                        # This check is implicitly handled by the size of dp_curr, but can add for clarity.
                        if new_length <= n: 
                            prev_sum = prev_s_offset - OFFSET
                            
                            # When adding `num` to a subsequence of length `prev_length`, it becomes
                            # the element at index `prev_length` (0-indexed) in the new subsequence of length `new_length`.
                            # The alternating sum term added is +num if prev_length is even, and -num if prev_length is odd.
                            sum_term = num if prev_length % 2 == 0 else -num
                            new_sum = prev_sum + sum_term
                            new_s_offset = new_sum + OFFSET
                            
                            # Check if the resulting sum is within the possible range
                            if 0 <= new_s_offset < SUM_RANGE_SIZE:
                                # Calculate potential new product by including `num`
                                potential_prod = -1 # Initialize as invalid product <= limit
                                
                                if prev_prod == 0 or num == 0:
                                    # If the previous product was 0 or the current number is 0, the new product is 0.
                                    potential_prod = 0
                                else: # previous product > 0 and current number > 0 (since nums[i] >= 0)
                                    # Check if multiplying `prev_prod` by `num` would exceed the `limit`.
                                    # Using integer division `limit // num` provides a safe check for `prev_prod * num <= limit`
                                    # for positive `num` without risking overflow if `prev_prod * num` is very large.
                                    if prev_prod <= limit // num: 
                                         potential_prod = prev_prod * num
                                    # Else: potential_prod > limit, it remains -1 (invalid for update)
                                
                                # If the calculated potential product is valid (>= 0 and <= limit)
                                if potential_prod != -1:
                                    # Update the state (new_length, new_s_offset) in the current DP table (dp_curr).
                                    # We take the maximum product found so far for this specific state.
                                    dp_curr[new_length][new_s_offset] = max(dp_curr[new_length][new_s_offset], potential_prod)
            
            # After processing the current number `nums[i]`, the `dp_curr` table contains
            # the maximum products for all possible states using elements from `nums[0...i]`.
            # This `dp_curr` table becomes the `dp_prev` table for the next iteration (considering nums[i+1]).
            dp_prev = dp_curr

        # After processing all numbers (from index 0 to n-1), the final `dp_prev` table
        # stores the maximum products for all possible subsequences of `nums`
        # with different lengths and alternating sums.
        
        max_overall_prod = -1 # Initialize the result to -1 (no valid subsequence found yet)
        
        # The target alternating sum is k. Convert it to the offsetted index for the DP table.
        target_s_offset = k + OFFSET
        
        # Check if the target sum k is within the possible range of sums that can be formed.
        # If not, no subsequence can have this sum.
        if 0 <= target_s_offset < SUM_RANGE_SIZE:
            # Iterate through all possible lengths for non-empty subsequences (from 1 to n).
            # The problem requires a non-empty subsequence.
            for length in range(1, n + 1):
                # Get the maximum product for subsequences of this length and the target sum k.
                current_prod = dp_prev[length][target_s_offset]
                
                # If a state with this length and target sum is reachable with a valid product (>= 0)
                if current_prod != -1:
                    # Update the overall maximum product found so far.
                    max_overall_prod = max(max_overall_prod, current_prod)

        # If no subsequence satisfies all the requirements (alternating sum k, non-empty, product <= limit),
        # max_overall_prod remains its initial value of -1.
        return max_overall_prod