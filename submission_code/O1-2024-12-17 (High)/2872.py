class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        curr_sum = 0
        max_val = 0
        # Traverse from right to left
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] <= curr_sum:
                curr_sum += nums[i]
            else:
                curr_sum = nums[i]
            max_val = max(max_val, curr_sum)
        return max_val