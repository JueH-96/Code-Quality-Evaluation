from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        """
        Finds the maximum number of elements that can be selected from the final array
        (after increasing elements by at most 1) such that the selected elements
        are consecutive when sorted.

        Args:
            nums: A list of positive integers.

        Returns:
            The maximum number of consecutive elements that can be selected.
        """
        # Sort the input array. This is crucial for the DP approach.
        nums.sort()
        
        # dp[v] stores the maximum length of a consecutive sequence ending at value v.
        # This length is achieved by selecting a subset of elements from the original
        # `nums` array processed so far (from index 0 up to the current element's index).
        # Each element from the original array can be used at most once to contribute
        # one value to the consecutive sequence.
        # The keys in the dictionary will be the possible final values, which are
        # elements from `nums` or `elements from nums` + 1.
        dp = {}
        
        # The minimum possible answer is 1 (selecting a single element and keeping it or incrementing it).
        # Initialize max_len to 0, it will be updated to at least 1 for any non-empty input.
        max_len = 0

        # Iterate through the sorted array.
        # Let the current element be `x`.
        # We consider how this element `x` can be used to contribute to extending consecutive sequences.
        # It can either be kept as `x` or incremented to `x + 1`.
        for x in nums:
            # Option 1: Use the current element `x` to form the value `x`.
            # If we use `x` to form the value `x` in a consecutive sequence,
            # this sequence must end at `x`. It can extend a sequence ending at `x - 1`.
            # The length of the sequence ending at `x` formed this way is the max length
            # ending at `x - 1` (using elements processed before or previous identical ones) plus 1.
            len_ending_at_x_from_x = dp.get(x - 1, 0) + 1
            
            # Option 2: Use the current element `x` to form the value `x + 1`.
            # If we use `x` to form the value `x + 1` in a consecutive sequence,
            # this sequence must end at `x + 1`. It can extend a sequence ending at `x`.
            # The length of the sequence ending at `x + 1` formed this way is the max length
            # ending at `x` (using elements processed before or previous identical ones) plus 1.
            len_ending_at_x_plus_1_from_x = dp.get(x, 0) + 1
            
            # Update the dp table.
            # For the value `x + 1`, the maximum length is the maximum of its current value
            # in the dp table (derived from elements processed before `x` or a previous
            # identical `x` transformed to `x+1`) and the new length derived from
            # using the current element `x` transformed to `x+1`.
            # It's important that `dp.get(x, 0)` uses the value before potentially
            # updating `dp[x]` in this iteration. By calculating `len_ending_at_x_plus_1_from_x`
            # first, and then updating `dp[x+1]`, we use the state from the previous iteration.
            dp[x + 1] = max(dp.get(x + 1, 0), len_ending_at_x_plus_1_from_x)
            
            # For the value `x`, the maximum length is the maximum of its current value
            # in the dp table and the new length derived from using the current element
            # `x` transformed to `x`.
            # This update for dp[x] uses dp.get(x-1, 0), which is not affected by
            # the calculations or updates for the current element `x` or `x+1`.
            dp[x] = max(dp.get(x, 0), len_ending_at_x_from_x)
            
            # The maximum length found so far is the maximum value in the dp table.
            # The maximum length can end at `x` or `x + 1` based on the current element's
            # contribution. We only need to check the values whose potential max length
            # could have increased due to the current element `x`.
            max_len = max(max_len, dp[x], dp[x + 1])

        # The loop finishes. max_len holds the maximum length of any consecutive sequence
        # that can be formed according to the rules. Since the constraints guarantee
        # at least one element in nums, max_len will be updated to at least 1.
        return max_len