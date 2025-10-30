class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        l = 0
        ans = 0
        for r in range(n):
            while nums[r] - nums[l] > 2 * k:
                l += 1
            ans = max(ans, r - l + 1)
        return ans