class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * n
        prefix_sum[0] = nums[0]
        for i in range(1, n):
            prefix_sum[i] = prefix_sum[i-1] + nums[i]
        for i in range(n-1, 1, -1):
            if prefix_sum[i-1] > nums[i]:
                return prefix_sum[i]
        return -1