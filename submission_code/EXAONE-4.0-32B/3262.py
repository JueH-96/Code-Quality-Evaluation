class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        total_sum = sum(nums)
        n = len(nums)
        for i in range(n - 1, -1, -1):
            if total_sum > 2 * nums[i]:
                return total_sum
            total_sum -= nums[i]
        return -1