import collections
from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Sort the array to process elements in increasing order.
        # This is crucial for the greedy strategy to work, as we aim to
        # pick the smallest possible distinct value for the current element,
        # ensuring it's greater than the previously chosen value, and
        # leaving maximum room for subsequent elements.
        nums.sort()

        distinct_count = 0
        # last_chosen_val keeps track of the largest value successfully chosen so far.
        # Initialize it to negative infinity to ensure the first element
        # can always pick a value that is considered "greater than" it.
        last_chosen_val = float('-inf')

        for num in nums:
            # Calculate the lower and upper bounds of the values 'num' can be transformed into.
            # An element 'x' can become any value 'v' such that x - k <= v <= x + k.
            low_bound = num - k
            high_bound = num + k

            # We want to find a target value for 'num' (let's call it 't_i') such that:
            # 1. t_i is within its allowed range [low_bound, high_bound].
            # 2. t_i is strictly greater than last_chosen_val (to ensure it's a new distinct element).
            # To maximize the total count of distinct elements, we should pick the smallest
            # possible t_i that satisfies both conditions. This leaves the largest possible
            # range for subsequent elements to pick their own distinct values.
            
            # The smallest value t_i can be is `low_bound`.
            # The smallest value strictly greater than `last_chosen_val` is `last_chosen_val + 1`.
            # Combining these, the most optimal candidate for a new distinct value is
            # `max(low_bound, last_chosen_val + 1)`.
            candidate_val = max(low_bound, last_chosen_val + 1)

            # Check if this candidate_val is within the permissible upper bound for the current 'num'.
            if candidate_val <= high_bound:
                # If it is, we have successfully found a new distinct value.
                distinct_count += 1
                last_chosen_val = candidate_val
            # If `candidate_val > high_bound`, it means even the smallest possible new distinct value
            # (`last_chosen_val + 1`) is too high for the current 'num' to reach, or its own
            # lower bound (`num - k`) is already too high to be chosen while being distinct from
            # previous elements. In this scenario, 'num' cannot contribute a new distinct value
            # that maintains our increasing sequence property. We effectively skip this 'num'.

        return distinct_count