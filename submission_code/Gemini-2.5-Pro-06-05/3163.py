from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        """
        Calculates the sum of the squares of distinct counts of all subarrays of nums.

        Args:
            nums: A 0-indexed integer array.

        Returns:
            The sum of the squares of the distinct counts of all subarrays.
        """
        
        n = len(nums)
        total_sum = 0
        
        # Iterate through each possible starting index of a subarray.
        for i in range(n):
            # This set will store the distinct elements for subarrays that start at index i.
            distinct_elements = set()
            
            # Iterate through each possible ending index of a subarray, starting from i.
            for j in range(i, n):
                # The current subarray under consideration is nums[i..j].
                # Add the element at the end of this subarray to our set.
                distinct_elements.add(nums[j])
                
                # The number of distinct elements in nums[i..j] is the current size of the set.
                distinct_count = len(distinct_elements)
                
                # Add the square of the distinct count to the running total.
                total_sum += distinct_count * distinct_count
                
        return total_sum