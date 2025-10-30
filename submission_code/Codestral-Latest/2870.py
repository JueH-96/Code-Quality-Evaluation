class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = nums.length
        max_len = -1
        i = 0

        while i < n - 1:
            if nums[i + 1] - nums[i] == 1:
                length = 2
                j = i + 2
                while j < n and nums[j] - nums[j - 1] == (-1) ** (j - i):
                    length += 1
                    j += 1
                max_len = max(max_len, length)
                i = j - 1
            else:
                i += 1

        return max_len