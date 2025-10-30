import math

class Solution:
    def minAnagramLength(self, s: str) -> int:
        counts = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1
        
        if not counts:
            return 0
            
        gcd_val = 0
        first = True
        for char in counts:
            if first:
                gcd_val = counts[char]
                first = False
            else:
                gcd_val = math.gcd(gcd_val, counts[char])
                
        min_len = 0
        for char in counts:
            min_len += counts[char] // gcd_val
            
        return min_len