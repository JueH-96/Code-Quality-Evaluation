class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        total_sum = sum(nums)
        
        # Try forming polygon starting with all numbers, removing largest if needed
        for i in range(len(nums) - 1, 1, -1):
            # Check if sum of all elements except the largest > largest element
            if total_sum - nums[i] > nums[i]:
                return total_sum
            # Remove the largest element for next iteration
            total_sum -= nums[i]
        
        # Less than 3 elements remaining, can't form polygon
        return -1