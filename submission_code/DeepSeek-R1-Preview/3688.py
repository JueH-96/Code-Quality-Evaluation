from collections import defaultdict
from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        # Function to compute max subarray, max prefix, max suffix, and total for a given array
        def compute_segment(arr):
            if not arr:
                return (float('-inf'), 0, 0, 0)
            max_sub = current = arr[0]
            max_prefix = current
            total = arr[0]
            for num in arr[1:]:
                current = max(num, current + num)
                max_sub = max(max_sub, current)
                max_prefix = max(max_prefix, total + num)
                total += num
            max_suffix = current
            current_total = 0
            max_suffix_total = float('-inf')
            for num in reversed(arr):
                current_total += num
                if current_total > max_suffix_total:
                    max_suffix_total = current_total
            return (max_sub, max_prefix, max_suffix_total, total)
        
        # Compute original maximum
        original_max = nums[0]
        current = nums[0]
        for num in nums[1:]:
            current = max(num, current + num)
            original_max = max(original_max, current)
        
        # Create a dictionary to map each x to its positions
        pos_dict = defaultdict(list)
        for idx, num in enumerate(nums):
            pos_dict[num].append(idx)
        
        max_sum = original_max
        
        # Process each x
        for x in pos_dict:
            positions = pos_dict[x]
            n = len(nums)
            # Split into segments
            segments = []
            prev = 0
            for p in positions:
                if prev <= p - 1:
                    segments.append(nums[prev:p])
                prev = p + 1
            if prev <= n - 1:
                segments.append(nums[prev:n])
            
            # Compute for each segment
            seg_info = []
            for seg in segments:
                if not seg:
                    continue
                max_sub, max_pre, max_suf, total = compute_segment(seg)
                seg_info.append((max_sub, max_pre, max_suf, total))
            
            if not seg_info:
                current_max = 0
            else:
                current_max = max(s[0] for s in seg_info)
                best_combined = 0
                best_prefix = 0
                for i, (s_max, s_pre, s_suf, s_total) in enumerate(seg_info):
                    if i == 0:
                        best_combined = s_suf
                        best_prefix = s_pre
                        current_max = max(current_max, s_max, best_combined)
                    else:
                        # Update current_max
                        current_max = max(current_max, best_combined + s_pre)
                        # Update best_combined
                        best_combined = max(best_combined + s_total, s_suf)
                        # Update best_prefix (if needed)
                        best_prefix = max(best_prefix + s_total, s_pre)
                        # Update current_max with best_combined
                        current_max = max(current_max, best_combined)
            
            if current_max > max_sum:
                max_sum = current_max
        
        return max_sum