class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        remainder_count = {}
        count = 0
        
        for hour in hours:
            remainder = hour % 24
            # We need to find numbers with remainder (24 - remainder) % 24
            target = (24 - remainder) % 24
            if target in remainder_count:
                count += remainder_count[target]
            
            # Add current remainder to the map
            remainder_count[remainder] = remainder_count.get(remainder, 0) + 1
        
        return count