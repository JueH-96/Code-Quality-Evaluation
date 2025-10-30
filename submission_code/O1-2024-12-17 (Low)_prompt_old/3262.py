class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the array in non-decreasing order
        nums.sort()
        
        # Compute the total sum of all sides
        total_sum = sum(nums)
        
        # Start from the largest index and keep removing the largest side
        # until we either form a valid polygon or run out of sides
        i = len(nums) - 1
        while i >= 2:  # Need at least 3 sides
            # Largest side in the current subset
            largest_side = nums[i]
            
            # Sum of other sides in current subset
            other_sum = total_sum - largest_side
            
            # If the sum of other sides is strictly greater than the largest side,
            # we can form a polygon
            if other_sum > largest_side:
                return total_sum  # Perimeter includes all sides in the current subset
            
            # Otherwise, remove the largest side and try again
            total_sum = other_sum
            i -= 1
        
        # If we exit the loop, no valid polygon can be formed
        return -1