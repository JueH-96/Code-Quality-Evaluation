from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int]) -> int:
        """
        Count the number of length-3 contiguous segments (in a circular sense)
        whose pattern is alternating: 0-1-0 or 1-0-1.
        """
        n = len(colors)
        cnt = 0
        
        for i in range(n):
            # indices of the 3-tile segment: i, i+1, i+2  (mod n)
            a = colors[i]
            b = colors[(i + 1) % n]
            c = colors[(i + 2) % n]
            
            if a == c and a != b:   # pattern 0-1-0 or 1-0-1
                cnt += 1
        
        return cnt