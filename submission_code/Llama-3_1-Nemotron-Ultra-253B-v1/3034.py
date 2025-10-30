from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        if not nums:
            return 0
        # Sort intervals based on the start time
        sorted_nums = sorted(nums, key=lambda x: x[0])
        merged = []
        for interval in sorted_nums:
            if not merged:
                merged.append(interval)
            else:
                last_start, last_end = merged[-1]
                current_start, current_end = interval
                if current_start <= last_end:
                    # Merge the intervals
                    new_end = max(last_end, current_end)
                    merged[-1][1] = new_end
                else:
                    merged.append(interval)
        # Calculate the total covered points
        total = 0
        for start, end in merged:
            total += end - start + 1
        return total