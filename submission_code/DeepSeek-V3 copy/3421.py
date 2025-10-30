from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        from collections import defaultdict
        
        # Create a frequency map of the remainders when divided by 24
        remainder_counts = defaultdict(int)
        count = 0
        
        for hour in hours:
            remainder = hour % 24
            # Find the complement that would make the sum a multiple of 24
            complement = (24 - remainder) % 24
            if complement in remainder_counts:
                count += remainder_counts[complement]
            # Update the frequency map
            remainder_counts[remainder] += 1
        
        return count