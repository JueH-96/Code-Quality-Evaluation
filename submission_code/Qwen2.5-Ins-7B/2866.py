class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        start = 0
        for i in range(len(nums)):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                start = i
                while i + 1 < len(nums) and nums[i] % 2 != nums[i + 1] % 2 and nums[i + 1] <= threshold:
                    i += 1
                max_length = max(max_length, i - start + 1)
            else:
                start = i + 1
        return max_length