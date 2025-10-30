class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the array in non-decreasing order
        nums.sort()
        
        # Iterate from the end of the sorted list to the start
        for i in range(len(nums) - 1, 1, -1):
            # Check if the current three numbers can form a triangle
            if nums[i] < nums[i - 1] + nums[i - 2]:
                # If they can, return the perimeter of the triangle
                return nums[i] + nums[i - 1] + nums[i - 2]
        
        # If no valid triangle is found, return -1
        return -1