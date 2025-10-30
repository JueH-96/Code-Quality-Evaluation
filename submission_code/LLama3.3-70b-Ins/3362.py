from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        # Initialize an empty list to store the uniqueness array
        uniqueness_array = []
        
        # Iterate over all possible subarrays
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                # Calculate the number of distinct elements in the current subarray
                distinct_elements = len(set(nums[i:j+1]))
                
                # Append the number of distinct elements to the uniqueness array
                uniqueness_array.append(distinct_elements)
        
        # Sort the uniqueness array in non-decreasing order
        uniqueness_array.sort()
        
        # Calculate the median of the uniqueness array
        n = len(uniqueness_array)
        median = uniqueness_array[n // 2] if n % 2 != 0 else uniqueness_array[n // 2 - 1]
        
        # Return the median of the uniqueness array
        return median