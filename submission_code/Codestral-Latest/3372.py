class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0

        max_length = 1
        current_length = 1

        for i in range(1, nums.length):
            if nums[i] > nums[i - 1]:
                current_length += 1
            elif nums[i] < nums[i - 1]:
                current_length += 1
            else:
                current_length = 1

            max_length = max(max_length, current_length)

        return max_length