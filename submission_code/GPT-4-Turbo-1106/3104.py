class Solution:
    def countWays(self, nums: list[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0
        ways = 0

        for i in range(n):
            if nums[i] >= n - i:
                count += 1
            else:
                ways += 1

        return ways + 1 if count == 0 else ways