from typing import List
from collections import Counter

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Map each number to the count of size-k subarrays it appears in
        # Using Counter from collections is convenient for accumulating counts.
        count_in_subarrays = Counter()
        
        # Iterate through all possible starting indices for a subarray of size k.
        # A subarray of size k starting at index i will end at index i + k - 1.
        # The starting index i can go from 0 up to n - k.
        for i in range(n - k + 1):
            # Get the current subarray (window) of size k.
            subarray = nums[i : i + k]
            
            # Find unique elements in the current subarray.
            # This is crucial because a number should only increment its count
            # for being present in a specific subarray *once*, even if it appears
            # multiple times within that single subarray.
            unique_elements_in_window = set(subarray)
            
            # For each unique element found in this window, increment its overall count
            # of appearances across all k-sized subarrays.
            for num in unique_elements_in_window:
                count_in_subarrays[num] += 1
        
        # Now, find the largest number that appeared in exactly one subarray of size k.
        largest_almost_missing = -1
        
        # Iterate through the collected counts for each number.
        for num, count in count_in_subarrays.items():
            # An integer is "almost missing" if its count is exactly 1.
            if count == 1:
                # We want the largest such integer. Update the maximum found so far.
                largest_almost_missing = max(largest_almost_missing, num)
                
        # Return the largest number that was found to be "almost missing".
        # If no number had a count of 1, largest_almost_missing will remain -1.
        return largest_almost_missing