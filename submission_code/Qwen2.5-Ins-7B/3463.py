from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        n = len(colors)
        count = 0
        i = 0
        while i < n:
            if i + 2 < n and colors[i] != colors[i + 2]:
                count += 1
                i += 3
            else:
                i += 1
        return count