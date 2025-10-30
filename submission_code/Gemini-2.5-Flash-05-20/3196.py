from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # 1. Sort the array to enable a sliding window approach
        nums.sort()

        # 2. Compute prefix sums for efficient range sum queries
        # prefix_sum[i] stores the sum of nums[0] to nums[i-1]
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]

        max_freq = 0
        left = 0 # Left pointer of the sliding window

        # 3. Iterate with the right pointer to expand the window
        for right in range(n):
            # For each window defined by [left...right], we want to find the cost
            # to make all elements equal to the median of this window.
            # The median index in the sorted array for the window [left...right]
            # is calculated relative to 'left' and then offset by 'left'.
            
            # The inner while loop shrinks the window from the left until it becomes valid
            # for the current 'right' position.
            while True:
                # Calculate the absolute index of the median element in nums_sorted
                # for the current window [left...right].
                # (right - left) // 2 gives the relative offset from 'left'.
                mid_idx = left + (right - left) // 2
                median_val = nums[mid_idx]

                # Calculate sum of elements in the left part of the window (including median)
                # This sum is from nums[left] to nums[mid_idx]
                sum_left_part = prefix_sum[mid_idx + 1] - prefix_sum[left]
                
                # Calculate sum of elements in the right part of the window (excluding median)
                # This sum is from nums[mid_idx + 1] to nums[right]
                sum_right_part = prefix_sum[right + 1] - prefix_sum[mid_idx + 1]

                # Number of elements in the left part (including median)
                count_left_part = mid_idx - left + 1
                
                # Number of elements in the right part (excluding median)
                count_right_part = right - mid_idx

                # Calculate the total operations needed to make all elements in the window
                # equal to median_val.
                # Cost for left part: sum(median_val - nums[p]) for p in [left, mid_idx]
                # Cost for right part: sum(nums[p] - median_val) for p in [mid_idx+1, right]
                cost = (count_left_part * median_val - sum_left_part) + \
                       (sum_right_part - count_right_part * median_val)
                
                if cost <= k:
                    # If the current window [left...right] is valid (cost within k),
                    # we can potentially achieve this frequency. Break from the inner loop
                    # and update max_freq. The outer loop will then try to expand 'right'.
                    break
                else:
                    # If the current window is too costly, we must shrink it from the left.
                    # Increment 'left' and recalculate the cost for the new, smaller window.
                    left += 1
            
            # After the inner while loop, [left...right] is a valid window.
            # Update the maximum frequency found so far.
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq