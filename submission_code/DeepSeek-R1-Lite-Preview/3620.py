from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Sort the array
        nums.sort()
        
        # Initialize previous value to negative infinity
        prev = -float('inf')
        count = 0
        
        # Iterate through each number in the sorted array
        for num in nums:
            # Calculate the target value
            target = max(prev + 1, num - k)
            # Check if target is within the allowable range
            if target <= num + k:
                # Assign this target value
                count += 1
                prev = target
            # Else, skip this element as it cannot be adjusted to a distinct value
        return count