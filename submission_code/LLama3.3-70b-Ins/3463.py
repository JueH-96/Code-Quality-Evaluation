from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        
        # Check for alternating groups in the array
        for i in range(n - 2):
            if colors[i] != colors[i + 1] and colors[i + 1] != colors[i + 2]:
                count += 1
        
        # Check for alternating group that wraps around the circle
        if colors[-1] != colors[0] and colors[0] != colors[1]:
            count += 1
        if colors[-2] != colors[-1] and colors[-1] != colors[0]:
            count += 1
        
        return count