class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n // 2
        while l < r:
            if nums[l] < nums[n - 1 - l]:
                l += 1
            else:
                break
        return n - 2 * l