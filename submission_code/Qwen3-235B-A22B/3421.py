from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        counts = [0] * 24
        for h in hours:
            counts[h % 24] += 1
        
        total = 0
        
        # Handle pairs with mod 0
        total += counts[0] * (counts[0] - 1) // 2
        
        # Handle pairs with mod 12
        total += counts[12] * (counts[12] - 1) // 2
        
        # Handle other pairs (i and 24 - i)
        for i in range(1, 12):
            complement = 24 - i
            total += counts[i] * counts[complement]
        
        return total