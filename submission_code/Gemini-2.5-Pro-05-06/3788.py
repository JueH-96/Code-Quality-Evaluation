import math

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Initialize with a very small number, as sums can be negative.
        # The problem implies nums is non-empty, so at least one element will be processed.
        overall_max_sum = -math.inf
        
        # Iterate over all possible non-empty subsequences S of nums.
        # A bitmask 'i' represents a subsequence: if j-th bit is set, nums[j] is in S.
        # Iterate i from 1 to (2^n - 1).
        for i in range(1, 1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1: # Check if j-th element is in the current subsequence
                    subsequence.append(nums[j])
            
            # Now, 'subsequence' is nums' (array after deletions).
            # It's guaranteed non-empty because i ranges from 1.
            m = len(subsequence)
            
            # Find max sum subarray of 'subsequence' with unique elements using sliding window.
            # This part takes O(m) time.
            
            max_sum_for_this_subsequence = -math.inf
            
            left = 0
            current_window_sum = 0
            seen_in_window = set()
            
            for right in range(m):
                val_at_right = subsequence[right]
                
                # Shrink window from left if val_at_right is already in seen_in_window
                while val_at_right in seen_in_window:
                    val_at_left = subsequence[left]
                    seen_in_window.remove(val_at_left)
                    current_window_sum -= val_at_left
                    left += 1
                
                # Expand window to include val_at_right
                seen_in_window.add(val_at_right)
                current_window_sum += val_at_right
                
                # Update max sum for the current subsequence S
                max_sum_for_this_subsequence = max(max_sum_for_this_subsequence, current_window_sum)
            
            # Update overall maximum sum
            # max_sum_for_this_subsequence could remain -math.inf if subsequence was empty,
            # but our loop for i ensures subsequence is non-empty.
            # A single element subsequence [x] yields max_sum_for_this_subsequence = x.
            if max_sum_for_this_subsequence != -math.inf : # Ensure a valid sum was found
                 overall_max_sum = max(overall_max_sum, max_sum_for_this_subsequence)
        
        return overall_max_sum