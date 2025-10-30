class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        lo, hi = 0, 1
        for hi in range(len(nums)):
            mid = (nums[lo] + nums[hi]) // 2
            if mid - nums[lo] <= k and nums[hi] - mid <= k:
                continue
            lo += 1
        return hi - lo + 1