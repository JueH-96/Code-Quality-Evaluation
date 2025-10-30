from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        current_len = 1
        expected_diff = 1  # The first required difference is +1
        
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i-1]
            if diff == expected_diff:
                current_len += 1
                expected_diff *= -1  # Flip the expected difference
                if current_len > max_len:
                    max_len = current_len
            else:
                if diff == 1:
                    # Start a new subarray with the current pair
                    current_len = 2
                    expected_diff = -1
                    if current_len > max_len:
                        max_len = current_len
                else:
                    # Reset the current length and expected difference
                    current_len = 1
                    expected_diff = 1
        
        return max_len if max_len >= 2 else -1