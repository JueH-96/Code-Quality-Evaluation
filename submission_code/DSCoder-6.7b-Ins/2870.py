class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = 0
        cur_len = 1
        for i in range(1, len(nums)):
            if nums[i] - nums[i-1] == 1:
                cur_len += 1
                max_len = max(max_len, cur_len)
            else:
                cur_len = 2
        return max_len if max_len > 1 else -1