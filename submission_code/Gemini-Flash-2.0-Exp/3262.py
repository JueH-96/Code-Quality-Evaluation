class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        perimeter = -1
        sum_so_far = 0
        for i in range(len(nums)):
            if i >= 2 and sum_so_far > nums[i]:
                perimeter = sum_so_far + nums[i]
            sum_so_far += nums[i]
        return perimeter