class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = 0
        for i in range(len(nums) - 1):
            if nums[i + 1] != nums[i] + 1:
                continue
            current_length = 2
            step_sign = -1
            j = i + 2
            while j < len(nums):
                if nums[j] - nums[j - 1] == step_sign:
                    current_length += 1
                    step_sign *= -1
                    j += 1
                else:
                    break
            if current_length > max_length:
                max_length = current_length
        return max_length if max_length >= 2 else -1