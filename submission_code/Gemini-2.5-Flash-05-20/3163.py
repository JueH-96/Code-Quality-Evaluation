from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum_of_squares = 0

        # Iterate through all possible starting indices `i`
        # `i` ranges from 0 to n-1
        for i in range(n):
            # For a fixed starting index `i`, we expand the subarray by increasing the ending index `j`.
            # `current_distinct_elements` will store the distinct numbers in the subarray nums[i...j].
            current_distinct_elements = set()

            # Iterate through all possible ending indices `j` for the current `i`.
            # `j` ranges from `i` to n-1.
            # This inner loop builds all subarrays starting at `i`.
            for j in range(i, n):
                # Add the current element `nums[j]` to the set.
                # Adding an element to a set takes O(1) on average. If the element is already present,
                # the set remains unchanged but the operation still completes in O(1).
                current_distinct_elements.add(nums[j])

                # The distinct count for the subarray `nums[i...j]` is the current size of the set.
                # Getting the length of a set takes O(1).
                distinct_count = len(current_distinct_elements)

                # Add the square of this distinct count to the total sum.
                # This contributes `distinct_count^2` for the subarray `nums[i...j]`.
                total_sum_of_squares += (distinct_count ** 2)

        # After iterating through all possible subarrays, `total_sum_of_squares` will hold the final result.
        return total_sum_of_squares