from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        
        # Calculate the initial sum of nums1
        current_sum = sum(nums1)
        
        # If the initial sum is already less than or equal to x, return 0
        if current_sum <= x:
            return 0
        
        # Create a list of tuples (nums2[i], nums1[i]) and sort by nums2[i] descending
        # This helps us prioritize setting to zero the elements with the largest increment rate
        increments = sorted(zip(nums2, nums1), reverse=True)
        
        # Try to reduce the sum by setting elements to zero
        for time in range(1, n + 1):
            # Set the element with the highest nums2[i] to zero
            current_sum -= increments[time - 1][1]
            
            # After setting to zero, the sum is incremented by nums2[i] for each element
            current_sum += sum(nums2)
            
            # Check if the current sum is less than or equal to x
            if current_sum <= x:
                return time
        
        # If we cannot achieve the target sum, return -1
        return -1