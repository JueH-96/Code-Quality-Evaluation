class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        for k in range(n, 2, -1):
            if prefix[k-1] > nums[k-1]:
                return prefix[k]
        return -1