class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        current_length = 1
        expected_diff = None
        
        for i in range(1, len(nums)):
            if current_length == 1:
                if nums[i] == nums[i-1] + 1:
                    current_length = 2
                    expected_diff = -1
                else:
                    current_length = 1
                    expected_diff = None
            else:
                if nums[i] - nums[i-1] == expected_diff:
                    current_length += 1
                    expected_diff *= -1
                else:
                    if nums[i] == nums[i-1] + 1:
                        current_length = 2
                        expected_diff = -1
                    else:
                        current_length = 1
                        expected_diff = None
            if current_length >= 2 and current_length > max_length:
                max_length = current_length
        return max_length if max_length >= 2 else -1