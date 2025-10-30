class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n

        left = 0
        right = n - 1
        removed_pairs = 0

        while left < right:
            if nums[left] < nums[right]:
                removed_pairs += 1
                left += 1
                right -= 1
            else:
                right -= 1

        min_length = n - 2 * removed_pairs
        return min_length