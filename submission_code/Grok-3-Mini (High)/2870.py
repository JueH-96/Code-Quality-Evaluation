from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = -1
        
        # Handle the first possible pair
        if n > 1 and nums[1] - nums[0] == 1:
            current_length = 2
            expected_diff = -1
            max_length = 2
        else:
            current_length = 1
            expected_diff = 1  # Arbitrary value, will be overwritten if needed
        
        # Iterate through the array starting from index 2
        for i in range(2, n):
            diff = nums[i] - nums[i - 1]
            if diff == expected_diff:
                current_length += 1
                expected_diff = -expected_diff  # Toggle the expected difference
                if current_length > max_length:
                    max_length = current_length
            else:
                if diff == 1:
                    # Start a new alternating subarray
                    current_length = 2
                    expected_diff = -1
                    if current_length > max_length:
                        max_length = current_length
                else:
                    # No alternating subarray, reset current length
                    current_length = 1
        
        return max_length