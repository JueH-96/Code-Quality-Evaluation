class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        @cache
        def helper(i: int, prev_diff: int) -> int:
            if i >= len(nums):
                return 0
            max_len = 0
            for j in range(i + 1, len(nums)):
                current_diff = abs(nums[j] - nums[i])
                if current_diff <= prev_diff:
                    max_len = max(
                        max_len, 1 + helper(j, current_diff))
            return max(max_len, 1)
        max_length = 0
        for i in range(len(nums) - 1):
            max_length = max(max_length, helper(i, 300))
        return max_length + 1