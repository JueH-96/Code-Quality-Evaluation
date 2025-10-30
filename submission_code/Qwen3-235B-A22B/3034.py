from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        if not nums:
            return 0
        
        # Sort the intervals based on their start points
        nums.sort()
        
        merged = []
        for start, end in nums:
            if not merged:
                merged.append([start, end])
            else:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    # Merge the intervals by updating the end of the last merged interval
                    merged[-1][1] = max(last_end, end)
                else:
                    merged.append([start, end])
        
        # Calculate the total number of covered points
        total = 0
        for s, e in merged:
            total += e - s + 1
        
        return total