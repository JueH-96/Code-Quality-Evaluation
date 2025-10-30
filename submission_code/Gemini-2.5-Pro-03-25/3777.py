import collections
from typing import List

class Solution:
  """
  Finds a non-empty subsequence of nums with alternating sum equal to k,
  maximizing the product of its numbers without exceeding limit.
  Returns the maximum product, or -1 if no such subsequence exists.
  
  The alternating sum of a 0-indexed array [s0, s1, s2, ...] is s0 - s1 + s2 - s3 + ...
  """
  def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
    """
    :param nums: List of integers. The elements are non-negative (0 <= nums[i] <= 12).
    :param k: Target alternating sum. Can be positive, negative or zero.
    :param limit: Maximum allowed product for the subsequence. limit >= 1.
    :return: Maximum product satisfying conditions, or -1 if no such subsequence exists.
    """

    # Dynamic Programming approach.
    # The state of our DP will be based on the alternating sum achieved.
    # For each possible alternating sum `s`, we need to track the maximum product achieved.
    # Since the sign of an element added to the alternating sum depends on its position (even/odd index)
    # in the subsequence, we need to track the maximum product separately for subsequences of even length
    # and odd length that result in the sum `s`.
    
    # dp state definition: dp[alt_sum] = [max_product_even_len, max_product_odd_len]
    # dp[s][0] stores the maximum product of a non-empty subsequence with alternating sum `s` and even length.
    # dp[s][1] stores the maximum product of a non-empty subsequence with alternating sum `s` and odd length.
    # We use collections.defaultdict initialized to return [-1, -1] for non-existent keys.
    # A value of -1 indicates that no valid subsequence with that sum and parity has been found yet,
    # or all subsequences found so far for that state exceeded the product limit.
    dp = collections.defaultdict(lambda: [-1, -1])

    # Iterate through each number `x` in the input list `nums`.
    for x in nums:
        # We use an 'updates' dictionary to store the potential new states or updates derived 
        # from including the current element 'x'. This approach avoids modifying 'dp' while iterating 
        # over it, which is crucial for correctness in DP state transitions.
        updates = collections.defaultdict(lambda: [-1, -1])

        # Option 1: Consider 'x' as the start of a new subsequence.
        # The subsequence `[x]` has length 1 (odd), alternating sum `x`, and product `x`.
        # This is only valid if the product `x` itself does not exceed the limit.
        # Since nums[i] >= 0, we have x >= 0. The check is simply `x <= limit`.
        if x <= limit:
            s_new = x  # Alternating sum is just x for a single element subsequence
            P_new = x  # Product is just x
            # Update the entry for odd length subsequences associated with sum s_new.
            # We use max() to ensure that if this state (s_new, odd_len) is potentially reached 
            # via multiple ways (e.g., starting new vs extending existing), we keep the max product.
            # In this step, updates[s_new][1] might already have a value if derived from extending 
            # an even length subsequence. We take the max to cover all possibilities.
            updates[s_new][1] = max(updates[s_new][1], P_new)

        # Option 2: Append 'x' to existing non-empty subsequences tracked in the 'dp' table.
        # Iterate through all existing states (sum `s`, [P_even, P_odd]) currently recorded in dp.
        # dp.items() provides pairs of (sum, [max_prod_even, max_prod_odd]).
        for s, P_vals in dp.items():
            P_even, P_odd = P_vals # Maximum products achieved for sum 's' with even/odd lengths respectively

            # Case A: Append 'x' to a subsequence that previously had an even length.
            # Original subsequence S has length L (even). New subsequence S + [x] has length L+1 (odd).
            # The new element 'x' is placed at index L (which is an even position, 0-based).
            # According to the alternating sum definition, elements at even indices are added.
            # New alternating sum = s + x.
            # The product is multiplied by x: New product = P_even * x.
            if P_even != -1: # Check if the even length state for sum 's' is valid (product > -1)
                s_new = s + x
                P_new = P_even * x
                # Check if the calculated product is within the specified limit.
                if P_new <= limit:
                    # Update the state for odd length subsequences with the new sum s_new.
                    # Use max() to ensure we store the largest product if this state is reached via multiple paths within this step.
                    updates[s_new][1] = max(updates[s_new][1], P_new)
            
            # Case B: Append 'x' to a subsequence that previously had an odd length.
            # Original subsequence S has length L (odd). New subsequence S + [x] has length L+1 (even).
            # The new element 'x' is placed at index L (which is an odd position, 0-based).
            # According to the alternating sum definition, elements at odd indices are subtracted.
            # New alternating sum = s - x.
            # The product is multiplied by x: New product = P_odd * x.
            if P_odd != -1: # Check if the odd length state for sum 's' is valid
                s_new = s - x
                P_new = P_odd * x
                 # Check if the calculated product is within the specified limit.
                if P_new <= limit:
                    # Update the state for even length subsequences with the new sum s_new.
                    # Use max() to keep the largest product.
                    updates[s_new][0] = max(updates[s_new][0], P_new)

        # Merge the collected updates into the main 'dp' table.
        # Iterate through all states (sums s_new) that were potentially created or modified in this step.
        for s_new, P_new_vals in updates.items():
            # Retrieve current maximum products stored in dp for s_new.
            # If s_new is encountered for the first time this loop iteration, dp[s_new] returns the default [-1, -1].
            current_P_even, current_P_odd = dp[s_new] 
            # Get the maximum products computed in this step for s_new from the 'updates' dictionary.
            updated_P_even, updated_P_odd = P_new_vals
            
            # Update the dp table entry for s_new by taking the element-wise maximum.
            # This ensures that dp[s_new] always stores the highest product achieved so far
            # for both even and odd length parities leading to sum s_new, considering paths from previous steps
            # and updates within the current step.
            dp[s_new] = [max(current_P_even, updated_P_even), max(current_P_odd, updated_P_odd)]

    # After iterating through all elements in 'nums', the 'dp' table contains the maximum products
    # for all achievable alternating sums and length parities.
    # We need to find the result for the target alternating sum 'k'.
    
    # Check if 'k' exists as a key in dp. This means 'k' is an achievable alternating sum
    # by at least one non-empty subsequence whose product is within the limit.
    if k in dp:
        # Retrieve the pair of maximum products [P_even, P_odd] associated with the target sum 'k'.
        result_pair = dp[k]
        # The final answer is the maximum product found for sum 'k', considering both possibilities
        # of the subsequence having an even or odd length.
        final_product = max(result_pair[0], result_pair[1])
        
        # If both entries in result_pair are -1, it means that although 'k' might be reachable,
        # all subsequences resulting in sum 'k' either had products exceeding the limit, or were
        # derived from intermediate states that were pruned due to the limit.
        # In this case, max() correctly returns -1.
        return final_product
    else:
        # If 'k' is not found as a key in the dp table after processing all numbers,
        # it signifies that no non-empty subsequence yielded the target alternating sum 'k'
        # while satisfying the product limit constraint.
        return -1