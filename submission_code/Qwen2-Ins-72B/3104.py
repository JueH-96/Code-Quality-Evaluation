class Solution:
    def countWays(self, nums: List[int]) -> int:
        nums.sort()
        ans = 0
        for k in range(1, len(nums) - 1):
            if nums[k - 1] < k < nums[k]:
                ans += 1
        return ans + (nums[0] > 0) + (nums[-1] < len(nums))