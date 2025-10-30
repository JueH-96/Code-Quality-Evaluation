class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        mid = (n + 1) // 2
        left = 0
        right = mid
        while right < n:
            if left < mid and nums[left] < nums[right]:
                left += 1
            right += 1
        return n - 2 * left