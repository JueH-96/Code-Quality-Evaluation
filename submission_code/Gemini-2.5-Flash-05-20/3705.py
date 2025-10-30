from collections import defaultdict
from typing import List

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # This dictionary will store how many distinct k-sized subarrays each number appears in.
        # Key: integer value from nums
        # Value: count of distinct k-sized subarrays containing this integer
        element_subarray_counts = defaultdict(int)

        n = len(nums)

        # Iterate through all possible starting indices for subarrays of size k
        # A subarray of size k starting at index 'i' goes up to index 'i + k - 1'.
        # The last possible starting index is n - k.
        for i in range(n - k + 1):
            # Extract the current subarray of size k
            current_subarray = nums[i : i + k]
            
            # Get unique elements within the current subarray.
            # Using a set ensures that if a number appears multiple times within 
            # the same subarray (e.g., [1, 5, 1] with k=3), it still only counts 
            # as "appearing in this one specific subarray".
            unique_elements_in_subarray = set(current_subarray)
            
            # For each unique element found in this current subarray, increment its global count
            # of k-sized subarrays it has appeared in.
            for element in unique_elements_in_subarray:
                element_subarray_counts[element] += 1
        
        # Now, find the largest integer that appeared in exactly one subarray of size k.
        largest_almost_missing = -1 # Initialize with -1, as numbers are non-negative (0 <= nums[i] <= 50)

        # Iterate through the counts of all elements
        for element, count in element_subarray_counts.items():
            # If an element appeared in exactly one k-sized subarray
            if count == 1:
                # Update largest_almost_missing if the current element is larger
                largest_almost_missing = max(largest_almost_missing, element)
                
        return largest_almost_missing