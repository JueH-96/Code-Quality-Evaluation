class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()  # Sort in non-decreasing order
        total_sum = sum(nums)
        
        # Start from the largest element and work backwards
        for i in range(len(nums) - 1, 1, -1):
            total_sum -= nums[i]  # Remove current element from sum
            
            # Check if current element can be the longest side of a valid polygon
            if nums[i] < total_sum:
                return total_sum + nums[i]  # Return the perimeter
        
        return -1  # No valid polygon possible