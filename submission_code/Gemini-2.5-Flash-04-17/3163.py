from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0

        # Iterate through all possible starting indices i
        for i in range(n):
            # Initialize a set to keep track of distinct elements for subarrays starting at i
            distinct_elements = set()
            # Iterate through all possible ending indices j for the current starting index i
            for j in range(i, n):
                # Add the current element nums[j] to the set of distinct elements
                # for the subarray nums[i..j] (which is nums[i..j-1] + nums[j])
                distinct_elements.add(nums[j])
                # The number of distinct elements in the current subarray nums[i..j]
                distinct_count = len(distinct_elements)
                # Add the square of the distinct count to the total sum
                total_sum += distinct_count * distinct_count

        return total_sum