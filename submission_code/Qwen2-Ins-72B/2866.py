class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        current_len = 0
        for i in range(len(nums)):
            if i == 0 or (nums[i-1] % 2 == nums[i] % 2) or (nums[i] % 2 != 0) or (nums[i] > threshold):
                current_len = 0
            else:
                current_len += 1
                max_len = max(max_len, current_len)
        return max_len