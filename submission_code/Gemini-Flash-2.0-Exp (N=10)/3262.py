class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        perimeter = -1
        for i in range(len(nums) - 1, 1, -1):
            if sum(nums[i-2:i]) > nums[i]:
                perimeter = sum(nums[i-2:i+1])
                break
        return perimeter