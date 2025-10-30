from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        if not nums:
            return 0
        # Sort intervals based on their start points
        intervals = sorted(nums, key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_start, last_end = merged[-1]
                current_start, current_end = interval
                if current_start <= last_end:
                    # Merge the intervals
                    merged[-1] = [last_start, max(last_end, current_end)]
                else:
                    merged.append(interval)
        # Calculate the total number of points covered
        total = 0
        for start, end in merged:
            total += end - start + 1
        return total