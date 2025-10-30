class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                curr_len = 2
                expected_diff = -1
                for j in range(i + 2, len(nums)):
                    if nums[j] - nums[j - 1] == expected_diff:
                        curr_len += 1
                        expected_diff *= -1
                    else:
                        break
                max_len = max(max_len, curr_len)
        return max_len