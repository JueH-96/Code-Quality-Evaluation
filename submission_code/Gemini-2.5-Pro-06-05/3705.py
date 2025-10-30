import collections
from typing import List

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        
        # A dictionary to store how many k-sized subarrays each number appears in.
        # Key: number, Value: count of subarrays.
        subarray_counts = collections.defaultdict(int)
        
        # Iterate through all possible starting positions of subarrays of size k.
        num_subarrays = len(nums) - k + 1
        for i in range(num_subarrays):
            # For each subarray, find its unique elements. We only care if a number
            # exists in the subarray, not its frequency within that subarray.
            window = nums[i : i + k]
            unique_elements_in_window = set(window)
            
            # For each unique number, increment its count of subarray appearances.
            for num in unique_elements_in_window:
                subarray_counts[num] += 1
                
        # Find the largest number that appeared in exactly one subarray.
        # Initialize result to -1, the default value if none is found.
        result = -1
        
        # Iterate through the numbers and their collected counts.
        for num, count in subarray_counts.items():
            # An "almost missing" integer is one that appeared in exactly one subarray.
            if count == 1:
                # Update the result if this number is the largest one found so far.
                result = max(result, num)
                
        return result