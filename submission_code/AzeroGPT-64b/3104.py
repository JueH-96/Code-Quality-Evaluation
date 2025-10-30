class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        total, left, right = 0, 0, nums[0]
        for i in range(1, n):
            if left < nums[i]:
                total += max(left + 1 - right, 0)
            left, right = max(left, nums[i - 1]), nums[i]
        total += max(left + 1 - right, 1)
        return total + (n < right)