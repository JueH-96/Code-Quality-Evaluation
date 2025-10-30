from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0

        # prefix_si[k] is True if nums[0...k-1] is strictly increasing.
        # k=0 corresponds to an empty prefix.
        # k=1 corresponds to the prefix [nums[0]].
        prefix_si = [True] * (n + 1)
        for i in range(n - 1):
            if nums[i] >= nums[i+1]:
                # If nums[i] >= nums[i+1], then any prefix ending at or after index (i+1)
                # (i.e., of length i+2 or more) is not strictly increasing.
                # So, prefix_si[k] for k >= i+2 must be False.
                for k in range(i + 2, n + 1):
                    prefix_si[k] = False
                break # Further checks on prefixes are unnecessary as they will also be non-increasing

        # suffix_si[k] is True if nums[k...n-1] is strictly increasing.
        # k=n corresponds to an empty suffix.
        # k=n-1 corresponds to the suffix [nums[n-1]].
        suffix_si = [True] * (n + 1)
        for i in range(n - 1, 0, -1): # Iterate from n-2 down to 0 (for the index `i-1`)
            if nums[i-1] >= nums[i]:
                # If nums[i-1] >= nums[i], then any suffix starting at or before index (i-1)
                # (i.e., suffix_si[k] for k <= i-1) is not strictly increasing.
                for k in range(i - 1, -1, -1):
                    suffix_si[k] = False
                break # Further checks on suffixes are unnecessary

        # Iterate through all possible subarrays nums[start_idx : end_idx + 1]
        for start_idx in range(n):
            for end_idx in range(start_idx, n):
                # The remaining array consists of:
                #   left_part = nums[0 : start_idx]
                #   right_part = nums[end_idx + 1 : n]

                # Condition 1: Check if left_part is strictly increasing
                # prefix_si[start_idx] checks nums[0...start_idx-1]
                if not prefix_si[start_idx]:
                    continue

                # Condition 2: Check if right_part is strictly increasing
                # suffix_si[end_idx + 1] checks nums[end_idx+1...n-1]
                if not suffix_si[end_idx + 1]:
                    continue
                
                # Condition 3: Check connection between left_part and right_part
                # This condition is only relevant if both left_part and right_part are non-empty.
                if start_idx > 0 and end_idx < n - 1:
                    # Last element of left_part: nums[start_idx - 1]
                    # First element of right_part: nums[end_idx + 1]
                    if nums[start_idx - 1] >= nums[end_idx + 1]:
                        continue
                
                # If all three conditions are met, this subarray is incremovable
                count += 1
                
        return count