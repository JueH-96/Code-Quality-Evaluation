from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        if n == 0:
            return 0
        
        original0 = original[0]
        current_low = -float('inf')
        current_high = float('inf')
        
        for i in range(n):
            delta = original[i] - original0
            u, v = bounds[i]
            a = u - delta
            b = v - delta
            
            current_low = max(current_low, a)
            current_high = min(current_high, b)
            
            if current_low > current_high:
                return 0
        
        if current_low > current_high:
            return 0
        else:
            return current_high - current_low + 1