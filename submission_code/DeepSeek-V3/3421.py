from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        from collections import defaultdict
        
        # Create a frequency map for the remainders when divided by 24
        remainder_counts = defaultdict(int)
        count = 0
        
        for hour in hours:
            remainder = hour % 24
            # If remainder is 0, it pairs with itself
            if remainder == 0:
                count += remainder_counts.get(0, 0)
            else:
                # Otherwise, it pairs with (24 - remainder)
                count += remainder_counts.get(24 - remainder, 0)
            # Update the frequency map
            remainder_counts[remainder] += 1
        
        return count