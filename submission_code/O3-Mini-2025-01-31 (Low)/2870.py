from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = -1
        i = 0
        while i < n - 1:
            # Check if current pair starts an alternating subarray.
            if nums[i+1] != nums[i] + 1:
                i += 1
                continue
            # We found a potential alternating subarray starting at index i.
            current_length = 2  # start with the valid pair: nums[i], nums[i+1]
            # The next expected difference is -1.
            expected_diff = -1
            j = i + 1
            while j < n - 1:
                # Check if the difference matches expected_diff.
                diff = nums[j+1] - nums[j]
                if diff == expected_diff:
                    current_length += 1
                    # Flip expected_diff for the next difference.
                    expected_diff *= -1
                    j += 1
                else:
                    break
            max_length = max(max_length, current_length)
            # Move to the next starting position.
            i += 1
        return max_length