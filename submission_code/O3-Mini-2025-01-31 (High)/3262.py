from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the list in ascending order.
        nums.sort()
        total = sum(nums)
        
        # While we have at least 3 sides to form a polygon:
        while len(nums) >= 3:
            # The largest side is the last element (since array is sorted in ascending order).
            largest = nums[-1]
            # Check if the sum of the other sides is greater than the largest side.
            if total - largest > largest:
                return total
            else:
                # Remove the largest side, update the total sum,
                # and then try again with the remaining sides.
                total -= largest
                nums.pop()
                
        return -1