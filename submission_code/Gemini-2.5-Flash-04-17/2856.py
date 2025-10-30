from typing import List
import collections

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)

        # 1. Calculate the total number of distinct elements in the entire array
        # Using a set is an efficient way to find distinct elements.
        total_distinct_set = set(nums)
        total_distinct_count = len(total_distinct_set)

        count = 0

        # 2. Iterate through all possible start indices (i) of a subarray
        # A subarray starts at index i.
        for i in range(n):
            # Use a set to keep track of distinct elements encountered
            # in the current subarray nums[i...j] that starts at index i.
            # This set is re-initialized for each new starting index i.
            current_distincts = set()

            # 3. Iterate through all possible end indices (j) of a subarray,
            # starting from the current start index i.
            # A subarray starting at i ends at index j. The subarray is nums[i:j+1].
            for j in range(i, n):
                # Add the element at the current end index j to the set.
                # Adding to a set is efficient and automatically handles duplicates
                # within the subarray nums[i...j].
                current_distincts.add(nums[j])

                # 4. Check if the number of distinct elements in the current subarray
                # (which is nums[i...j], represented by the 'current_distincts' set)
                # is equal to the total number of distinct elements in the whole array.
                if len(current_distincts) == total_distinct_count:
                    # If the condition is met, this subarray nums[i...j] is "complete".
                    # Since we are iterating through every possible contiguous subarray [i, j]
                    # exactly once and checking the condition, we simply increment the total count
                    # of complete subarrays found so far.
                    count += 1
                    # Optimization: For a fixed start 'i', once we find an end 'j'
                    # such that nums[i:j+1] is complete, any subarray nums[i:k+1]
                    # with k > j will also be complete (because it contains all elements
                    # of nums[i:j+1] plus potentially more).
                    # However, the current O(N^2) structure implicitly counts these.
                    # The inner loop continues, and for j+1, j+2, etc., the set will
                    # still have the same number of distinct elements (or more),
                    # satisfying the condition again.
                    # So, no break is needed here for correctness, and the O(N^2) logic
                    # naturally counts all valid subarrays.

        # 5. After checking all possible subarrays, return the total count of complete subarrays.
        return count