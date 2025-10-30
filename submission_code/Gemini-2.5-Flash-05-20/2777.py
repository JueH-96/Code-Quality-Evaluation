from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = [0] * n

        # Step 1: Precompute suffix distinct counts
        # suffix_distinct_counts[k] will store the number of distinct elements
        # in the subarray nums[k...n-1].
        # We need size n+1 to handle the case of an empty suffix (index n).
        suffix_distinct_counts = [0] * (n + 1)
        current_suffix_set = set()
        
        # Iterate from the end of the array backwards to populate suffix_distinct_counts
        for k in range(n - 1, -1, -1):
            current_suffix_set.add(nums[k])
            suffix_distinct_counts[k] = len(current_suffix_set)
        
        # After this loop, suffix_distinct_counts[n] remains 0, which correctly
        # represents the number of distinct elements in an empty suffix.

        # Step 2: Calculate diff array by iterating through prefixes
        prefix_set = set()
        for i in range(n):
            # Add the current element to the prefix set
            prefix_set.add(nums[i])
            # The number of distinct elements in the prefix nums[0...i]
            prefix_distinct_count = len(prefix_set)

            # The number of distinct elements in the suffix nums[i+1...n-1]
            # This is found by looking up suffix_distinct_counts at index (i+1).
            # If i is n-1, then i+1 is n, and suffix_distinct_counts[n] is 0,
            # which is correct for an empty suffix.
            suffix_distinct_count = suffix_distinct_counts[i + 1]

            # Calculate the difference for the current index i
            diff[i] = prefix_distinct_count - suffix_distinct_count

        return diff