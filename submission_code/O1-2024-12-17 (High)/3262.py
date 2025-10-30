class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        total_sum = sum(nums)
        
        # We'll try to use as many elements as possible from the largest to the smallest
        for i in range(len(nums) - 1, 1, -1):
            largest = nums[i]
            if total_sum - largest > largest:
                # We can form a polygon with all current elements
                return total_sum
            # Otherwise, remove the largest element and keep checking
            total_sum -= largest
        
        # If we exit the loop, forming a polygon was not possible
        return -1