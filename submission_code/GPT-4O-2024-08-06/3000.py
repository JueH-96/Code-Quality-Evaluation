from typing import List
from sortedcontainers import SortedList

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        # Initialize the minimum difference to a large number
        min_diff = float('inf')
        
        # SortedList to maintain a sorted order of elements
        sorted_list = SortedList()
        
        # Iterate over the array
        for i in range(len(nums)):
            # Add the current element to the sorted list
            sorted_list.add(nums[i])
            
            # Ensure we only consider elements at least x indices apart
            if i >= x:
                # Remove the element that is no longer at least x indices apart
                sorted_list.remove(nums[i - x])
                
                # Find the position where nums[i] would fit in the sorted list
                pos = sorted_list.bisect_left(nums[i])
                
                # Check the closest element on the left if it exists
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos - 1]))
                
                # Check the closest element on the right if it exists
                if pos < len(sorted_list):
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos]))
        
        return min_diff