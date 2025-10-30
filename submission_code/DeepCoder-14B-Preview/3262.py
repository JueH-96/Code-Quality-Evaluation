class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        for k in range(n, 2, -1):
            if prefix[k - 1] > nums[k - 1]:
                return prefix[k]
        return -1