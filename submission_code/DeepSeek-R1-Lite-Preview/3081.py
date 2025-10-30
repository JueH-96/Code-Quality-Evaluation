class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        right = 0
        count = 0
        while left < n and right < n:
            if nums[left] < nums[right]:
                count += 1
                left += 1
                right += 1
            else:
                right += 1
        return max(n - 2 * count, 0)