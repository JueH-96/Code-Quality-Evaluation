import collections
from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        freq = collections.Counter()
        min_idx_dict = {}
        max_idx_dict = {}
        for i, num in enumerate(nums):
            if num not in min_idx_dict:
                min_idx_dict[num] = i
            max_idx_dict[num] = i  # Set to current index; will be overwritten to last index
            freq[num] += 1
        
        intervals = []
        for num in freq:
            if freq[num] >= 2:
                min_idx_val = min_idx_dict[num]
                max_idx_val = max_idx_dict[num]
                start = min_idx_val
                end = max_idx_val - 1
                intervals.append((start, end))
        
        if not intervals:
            forbidden_count = 0
        else:
            intervals.sort()  # Sort intervals by start index
            current_start = intervals[0][0]
            current_end = intervals[0][1]
            forbidden_count = 0
            for i in range(1, len(intervals)):
                next_start, next_end = intervals[i][0], intervals[i][1]
                if next_start <= current_end + 1:  # Overlap or adjacent
                    current_end = max(current_end, next_end)
                else:
                    # Add length of current merged interval
                    forbidden_count += current_end - current_start + 1
                    current_start = next_start
                    current_end = next_end
            # Add length of the last merged interval
            forbidden_count += current_end - current_start + 1
        
        total_cuts = n - 1
        k = total_cuts - forbidden_count
        MOD = 1000000007
        ans = pow(2, k, MOD)
        return ans