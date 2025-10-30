from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = [0] * n

        # Calculate distinct elements in prefixes
        # Store the count of distinct elements in prefix nums[0...i] at index i
        pref_distinct_counts = [0] * n
        prefix_set = set()
        for i in range(n):
            prefix_set.add(nums[i])
            pref_distinct_counts[i] = len(prefix_set)

        # Calculate distinct elements in suffixes and compute the difference
        suffix_set = set()
        # Iterate from right to left
        for i in range(n - 1, -1, -1):
            # The suffix for index i is nums[i+1 ... n-1].
            # When iterating from right to left, the 'suffix_set' at index i
            # contains elements from nums[i+1] to nums[n-1] because nums[i]
            # is added *after* calculating the distinct count for the suffix starting at i+1.
            suffix_distinct_count = len(suffix_set)

            # The prefix for index i is nums[0 ... i].
            # Its distinct count was calculated in the first pass.
            prefix_distinct_count = pref_distinct_counts[i]

            # Calculate the difference for index i
            diff[i] = prefix_distinct_count - suffix_distinct_count

            # Add the current element nums[i] to the suffix_set.
            # This prepares the set for the next iteration (index i-1),
            # where nums[i] will be part of the suffix nums[i ... n-1].
            suffix_set.add(nums[i])

        return diff