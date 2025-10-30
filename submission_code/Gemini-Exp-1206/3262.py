class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = 0
        ans = -1
        for i in range(n):
            if i >= 2 and prefix_sum > nums[i]:
                ans = prefix_sum + nums[i]
            prefix_sum += nums[i]
        return ans