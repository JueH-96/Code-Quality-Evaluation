from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # Array to keep how many times each remainder (0..23) has appeared so far
        freq = [0] * 24
        
        pairs = 0
        for h in hours:
            r = h % 24                     # remainder when divided by 24
            comp = (24 - r) % 24           # required complement remainder
            
            # All previously seen values with remainder `comp`
            # form a valid pair with the current value
            pairs += freq[comp]
            
            # Record the current remainder for future pairs
            freq[r] += 1
        
        return pairs