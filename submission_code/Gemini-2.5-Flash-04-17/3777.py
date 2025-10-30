from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        n = len(nums)
        
        # The maximum possible alternating sum magnitude for a subsequence of length j
        # is ceil(j/2) * max(nums_i). Max j is N. Max sum approx ceil(N/2) * 12.
        # For N=150, max sum approx 75 * 12 = 900.
        # The minimum possible alternating sum for a subsequence of length j
        # is -floor(j/2) * max(nums_i). Max j is N. Min sum approx -75 * 12 = -900.
        # Sum range is roughly [-900, 900].
        # We need an offset to map sums to non-negative array indices.
        # Let OFFSET = 900. Sum s maps to index s + OFFSET.
        # Index range [ -900 + 900, 900 + 900 ] = [0, 1800].
        # DP array size for sum dimension is 1801.
        
        # A slightly safer offset based on maximum possible sum change per element
        # Max change per element is 12. Max length is 150. Max sum difference from 0 is less than 150 * 12.
        # More accurately, max sum is bounded by ceil(N/2)*12 and min sum by -floor(N/2)*12.
        # For N=150, this is [-900, 900]. OFFSET = 900 is correct.
        OFFSET = 900
        SUM_RANGE = 2 * OFFSET + 1 # 1801

        # dp[j][s_idx]: max product of a subsequence of length j with sum s=s_idx-OFFSET
        # after processing elements up to index i-1.
        # Use two layers: prev_dp and curr_dp

        # Initialize dp table for processing 0 elements (empty prefix)
        # dp[0] stores states after processing 0 elements
        dp = [[-1] * SUM_RANGE for _ in range(n + 1)]
        
        # Base case: empty subsequence (length 0, sum 0, product 1)
        # This is the only valid state before processing any elements.
        dp[0][OFFSET] = 1

        # Iterate through nums (processing nums[i] one by one)
        for i in range(n):
            num = nums[i]
            
            # Create the DP table for results after processing nums[0...i]
            # Initialize it by copying the results from processing nums[0...i-1] (stored in `dp`)
            # This accounts for subsequences that *exclude* nums[i].
            curr_dp = [[-1] * SUM_RANGE for _ in range(n + 1)]
            for j in range(n + 1): # Length
                 for s_idx in range(SUM_RANGE): # Sum index
                     curr_dp[j][s_idx] = dp[j][s_idx]

            # Now, consider subsequences that *include* nums[i].
            # If we include nums[i], it must be appended to a subsequence formed using nums[0...i-1].
            # Let the previous subsequence have length `prev_j` and sum `prev_s`.
            # This previous subsequence is found in the `dp` table (results before processing nums[i]).
            # The new subsequence will have length `j = prev_j + 1`.
            # nums[i] is appended at index `prev_j` within the new subsequence.
            # Its contribution to the sum is `(-1)^prev_j * nums[i]`.
            # The new sum `s = prev_s + (-1)^prev_j * nums[i]`.

            # Iterate through possible *previous* lengths `prev_j`.
            # `prev_j` can be any length achievable using `nums[0...i-1]`, from 0 up to `i`.
            for prev_j in range(i + 1):
                 # The new length `j` after including `num` is `prev_j + 1`
                 j = prev_j + 1
                 
                 # Check if the new length `j` is valid (<= N)
                 if j > n: 
                     continue # Should not happen with correct loop bounds

                 # Iterate through possible previous sum indices `prev_s_idx` in the `dp` table (from the previous step)
                 for prev_s_idx in range(SUM_RANGE):
                      # Check if the previous state (length prev_j, sum prev_s = prev_s_idx - OFFSET) was reachable
                      if dp[prev_j][prev_s_idx] != -1:
                           prev_s = prev_s_idx - OFFSET # Previous sum
                           prev_prod = dp[prev_j][prev_s_idx]

                           # Calculate the new sum `s` by including `num` at index `prev_j` in the subsequence
                           # The index of `num` in the new subsequence is `prev_j`.
                           sum_change = num if prev_j % 2 == 0 else -num
                           s = prev_s + sum_change
                           s_idx = s + OFFSET

                           # Check if the new sum index is valid (within the bounds of SUM_RANGE)
                           if 0 <= s_idx < SUM_RANGE:
                                current_prod = prev_prod * num

                                # Check product limit
                                if current_prod <= limit:
                                     # Update the current state `curr_dp[j][s_idx]`
                                     # We take the maximum product found so far for this state.
                                     curr_dp[j][s_idx] = max(curr_dp[j][s_idx], current_prod)

            # After computing all states for subsequences using nums[0...i],
            # the curr_dp table becomes the dp table for the next iteration.
            dp = curr_dp

        # After iterating through all numbers (nums[0...N-1]), dp contains results using all nums
        # Find the maximum product among all non-empty subsequences (j > 0)
        # that have the target alternating sum k and product <= limit.
        
        max_overall_product = -1
        target_s_idx = k + OFFSET

        # Check if the target sum index is within the computed range
        if 0 <= target_s_idx < SUM_RANGE:
            # Iterate through possible lengths j (non-empty, so j > 0)
            for j in range(1, n + 1):
                # Get the maximum product for subsequences of length j and target sum k
                product = dp[j][target_s_idx]
                
                # If a valid subsequence exists with this length and sum (product is not -1)
                if product != -1:
                    # Since we checked limit during DP update, this product is valid.
                    # Update the overall maximum product.
                    max_overall_product = max(max_overall_product, product)

        return max_overall_product