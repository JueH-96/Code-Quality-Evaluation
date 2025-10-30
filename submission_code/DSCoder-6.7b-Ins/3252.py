class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        i = 0
        while i < n:
            j = i + 1
            while j < n and nums[j] > nums[j-1]:
                j += 1
            if i > 0 and j < n - 1:
                count += (i + 1) * (n - j)
            elif i > 0 or j < n - 1:
                count += 1
            i = j
        return count