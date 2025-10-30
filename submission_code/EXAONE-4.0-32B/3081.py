class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        right = n // 2
        pairs = 0
        while left < n // 2 and right < n:
            if nums[left] < nums[right]:
                pairs += 1
                left += 1
                right += 1
            else:
                right += 1
        return n - 2 * pairs