class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        from collections import defaultdict
        
        # Dictionary to store the frequency of remainders when divided by 24
        remainder_count = defaultdict(int)
        
        # Count the number of pairs
        pairs_count = 0
        
        # Iterate over each hour in the list
        for hour in hours:
            # Calculate remainder when hour is divided by 24
            remainder = hour % 24
            
            # To form a complete day, we need another number which, when added
            # to this remainder, sums up to a multiple of 24.
            # We need a complement that, when added to remainder, gives 24 or 0 (mod 24)
            if remainder == 0:
                # If remainder is 0, we need another 0 for a complete day
                pairs_count += remainder_count[0]
            else:
                # Otherwise, we need (24 - remainder)
                complement = 24 - remainder
                pairs_count += remainder_count[complement]
            
            # Increment the count of this remainder
            remainder_count[remainder] += 1
        
        return pairs_count