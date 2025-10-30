class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        if nums[0] > 0:
            ans += 1
        for i in range(n):
            if i + 1 > nums[i]:
                if i == n - 1 or i + 1 < nums[i + 1]:
                    ans += 1
        return ans