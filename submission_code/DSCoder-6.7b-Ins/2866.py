class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0
        l = 0
        for r in range(n):
            if nums[r] > threshold:
                l = r + 1
            elif r > 0 and nums[r] % 2 == nums[r - 1] % 2:
                l = r
            while r - l + 1 > max_len and l > 0 and nums[l - 1] % 2 == 0:
                if nums[l - 1] <= threshold:
                    max_len = max(max_len, r - l + 1)
                l += 1
        return max_len