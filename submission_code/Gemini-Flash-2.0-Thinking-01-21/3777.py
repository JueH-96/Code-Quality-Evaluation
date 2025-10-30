from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        n = len(nums)
        
        # The maximum possible absolute value of the alternating sum
        # is bounded by n * max(nums[i]).
        max_possible_sum_abs = n * 12 # Using the upper bound of nums[i]
        
        # Offset the sum to handle negative values. Sum range will be [0, 2 * max_possible_sum_abs].
        offset = max_possible_sum_abs
        sum_range = 2 * max_possible_sum_abs + 1

        # dp[i][sum_offset][len_parity][is_non_empty] stores max product using subsequence from nums[0..i-1]
        # i: index in nums (0 to n)
        # sum_offset: alternating sum + offset (0 to sum_range - 1)
        # len_parity: length parity of subsequence (0 for even, 1 for odd) *before* considering nums[i]
        # is_non_empty: 0 for empty subsequence, 1 for non-empty
        
        # Initialize dp table with -1 (impossible state)
        # dp[i][sum_offset][len_parity][is_non_empty]
        dp = [[ [[-1 for _ in range(2)] for _ in range(2)] for _ in range(sum_range)] for _ in range(n + 1)]

        # Base case: Before processing any element (i=0), we have an empty subsequence.
        # Sum = 0, Length = 0 (even parity), Product = 1. It is empty (is_non_empty = 0).
        dp[0][offset][0][0] = 1

        # Iterate through nums elements
        for i in range(n):
            val = nums[i]

            # Iterate through all possible states achieved using nums[0..i-1]
            for sum_offset in range(sum_range):
                for len_parity in range(2): # len_parity is the length parity of the subsequence from nums[0..i-1]
                    for is_non_empty in range(2):
                        prev_prod = dp[i][sum_offset][len_parity][is_non_empty]

                        # If this state was not reachable, skip
                        if prev_prod == -1:
                            continue

                        # Option 1: Don't include nums[i] in the subsequence
                        # The subsequence properties (sum, length parity, product, non-empty status)
                        # remain the same as the one formed using nums[0..i-1].
                        # This state is now achievable using nums[0..i].
                        dp[i+1][sum_offset][len_parity][is_non_empty] = max(dp[i+1][sum_offset][len_parity][is_non_empty], prev_prod)

                        # Option 2: Include nums[i] in the subsequence
                        # This forms a new subsequence by appending nums[i] to the subsequence
                        # represented by the state dp[i][sum_offset][len_parity][is_non_empty].
                        # The new subsequence will always be non-empty.
                        new_is_non_empty = 1 

                        current_sum = sum_offset - offset
                        
                        # Calculate the contribution of nums[i] based on the length parity (len_parity)
                        # of the subsequence *before* adding nums[i].
                        sum_change = val if len_parity == 0 else -val
                        
                        new_sum = current_sum + sum_change
                        new_sum_offset = new_sum + offset
                        
                        # The new length parity is the opposite of the previous one.
                        new_len_parity = 1 - len_parity 

                        # Check if the new sum is within the valid offsetted range
                        if 0 <= new_sum_offset < sum_range:
                            
                            # Calculate the new product. This depends on the previous product and val.
                            # Initialize new_prod to an invalid value
                            new_prod = -1 

                            # If the previous state was the base empty subsequence (is_non_empty=0, prod=1)
                            # Adding val creates the subsequence [val]. Product is val.
                            # We check is_non_empty and prev_prod for the base case identifier.
                            # We don't need to check sum_offset and len_parity explicitly because dp[0][offset][0][0] is the ONLY state
                            # where is_non_empty is 0 and prev_prod is 1.
                            if is_non_empty == 0 and prev_prod == 1:
                                new_prod = val 
                            # Otherwise (the previous state represents a non-empty subsequence, or a state with prod 0).
                            else:
                                # If previous product was 0 or the current value is 0, the new product is 0.
                                if prev_prod == 0 or val == 0:
                                    new_prod = 0
                                # If previous product > 0 and current value > 0, multiply.
                                else: # prev_prod > 0 and val > 0
                                    new_prod = prev_prod * val

                            # Check if the calculated new product is valid (>= 0) and within the limit
                            if new_prod != -1 and new_prod <= limit:
                                # Update the state at i+1 for the new sum, new length parity,
                                # and specifically for non-empty subsequences.
                                dp[i+1][new_sum_offset][new_len_parity][1] = max(dp[i+1][new_sum_offset][new_len_parity][1], new_prod)


        # After iterating through all elements, the results for subsequences using nums[0..n-1] are in dp[n].
        # We are looking for a non-empty subsequence (is_non_empty = 1)
        # with alternating sum exactly k.
        
        target_sum_offset = k + offset
        
        # Check if the target sum k is within the range that can be represented by offsetted sums.
        # If not, no subsequence achievable by the given numbers can reach this sum.
        if not (0 <= target_sum_offset < sum_range):
            return -1
            
        # The result is the maximum product found for the target sum k,
        # considering both even and odd length subsequences (length parity 0 or 1),
        # but only from the non-empty subsequence states (is_non_empty = 1).
        result = max(dp[n][target_sum_offset][0][1], dp[n][target_sum_offset][1][1])

        # If the result is still -1, it means no non-empty subsequence
        # was found that satisfies the conditions (sum k and product <= limit).
        return result