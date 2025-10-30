from sortedcontainers import SortedList
from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Transform nums to a new array where nums[i] becomes nums[i] - i
        transformed = [(nums[i] - i, i) for i in range(n)]
        
        # Sort the transformed array by the first element (nums[i] - i)
        transformed.sort()
        
        # Initialize a sorted list to keep track of the maximum sums
        max_sums = SortedList([(float('-inf'), -1)])
        
        # Initialize the result with the maximum element in nums
        result = max(nums)
        
        for value, index in transformed:
            # Find the position to insert the current value in the sorted list
            pos = max_sums.bisect_right((value, index))
            
            # The maximum sum ending at the current index
            current_max_sum = value + index + max_sums[pos - 1][0]
            
            # Update the result with the current maximum sum
            result = max(result, current_max_sum)
            
            # Insert the current maximum sum into the sorted list
            max_sums.add((current_max_sum - index, index))
        
        return result