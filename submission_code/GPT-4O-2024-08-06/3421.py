class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        from collections import defaultdict
        
        # Dictionary to store the frequency of each remainder when divided by 24
        remainder_count = defaultdict(int)
        
        # Count the number of pairs
        count = 0
        
        for hour in hours:
            remainder = hour % 24
            # Find the complement remainder that would sum to a multiple of 24
            complement = (24 - remainder) % 24
            
            # Add the number of hours that can pair with the current hour to form a complete day
            count += remainder_count[complement]
            
            # Update the remainder count for the current hour
            remainder_count[remainder] += 1
        
        return count