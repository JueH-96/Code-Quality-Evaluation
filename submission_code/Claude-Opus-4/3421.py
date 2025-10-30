class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # Count frequency of remainders when divided by 24
        remainder_count = {}
        count = 0
        
        for hour in hours:
            remainder = hour % 24
            
            # Check for pairs with current element
            # We need the complement that makes sum divisible by 24
            if remainder == 0:
                # If remainder is 0, we need another 0
                if 0 in remainder_count:
                    count += remainder_count[0]
            else:
                # Otherwise, we need 24 - remainder
                complement = 24 - remainder
                if complement in remainder_count:
                    count += remainder_count[complement]
            
            # Add current remainder to the map
            remainder_count[remainder] = remainder_count.get(remainder, 0) + 1
        
        return count