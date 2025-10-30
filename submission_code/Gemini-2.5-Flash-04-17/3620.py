from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Sort the array to process elements in non-decreasing order.
        # This allows us to greedily assign the smallest available distinct values,
        # which is optimal for maximizing the total number of distinct elements.
        nums.sort()

        # Initialize the last assigned value to a very small number.
        # This serves as a sentinel representing that no value has been assigned yet.
        # The specific value float('-inf') works because any real number (which
        # includes all possible results of n +/- k) is greater than float('-inf').
        # We need to use a value smaller than any possible n - k.
        # Since nums[i] >= 1 and k >= 0, the smallest possible n - k is 1 - 10^9.
        # float('-inf') works correctly for this purpose in comparisons.
        last_assigned = float('-inf')

        # Initialize the count of distinct elements found so far.
        distinct_count = 0

        # Iterate through the sorted numbers.
        for n in nums:
            # The smallest value we would ideally want to assign to the current number 'n'
            # to create a new distinct value is one greater than the last value assigned.
            # This ensures that the new value is strictly greater than the previous one,
            # maintaining the distinctness in increasing order.
            # If last_assigned is float('-inf'), required_val will effectively be a very large negative number,
            # smaller than any possible integer value for n - k.
            required_val = last_assigned + 1

            # The range of possible values that the current number 'n' can be transformed into
            # by adding an integer in [-k, k] is [n - k, n + k].
            min_val = n - k
            max_val = n + k

            # We need to assign 'n' to a value 't' such that:
            # 1. t is in the allowed range [min_val, max_val].
            # 2. t is strictly greater than last_assigned (i.e., t >= required_val).
            # To maximize the chance of finding distinct values for subsequent elements
            # (which are >= the current element n), we should choose the smallest possible
            # value 't' that satisfies both conditions.
            # The smallest value 't' that is >= required_val is required_val itself.
            # The smallest value 't' that is >= min_val is min_val itself.
            # Therefore, the smallest value 't' that is >= required_val AND >= min_val
            # is max(required_val, min_val). This is our candidate for the new distinct value.
            candidate_val = max(required_val, min_val)

            # Now we check if this candidate value is actually achievable within the allowed range [min_val, max_val] for 'n'.
            # By definition of max(), candidate_val is already >= min_val.
            # So, we only need to check if candidate_val is less than or equal to max_val.
            if candidate_val <= max_val:
                # If candidate_val is within the allowed range, we can transform 'n' into candidate_val.
                # This value is guaranteed to be greater than last_assigned (because candidate_val >= required_val = last_assigned + 1).
                # Thus, candidate_val is a new distinct value.
                last_assigned = candidate_val
                # Increment the count of distinct elements found.
                distinct_count += 1
            # If candidate_val > max_val, it means that even the smallest possible value
            # we could assign to 'n' that is greater than the last assigned value
            # is still larger than the maximum value 'n' can become (n + k).
            # In this situation, 'n' cannot be used to create a new distinct value
            # that extends the sequence of distinct values found so far.
            # We simply move to the next element in the sorted array.
            # last_assigned and distinct_count remain unchanged.

        # After iterating through all elements, distinct_count holds the maximum possible
        # number of distinct elements achievable.
        return distinct_count