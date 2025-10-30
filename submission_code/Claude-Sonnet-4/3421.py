class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # Count frequency of each remainder when divided by 24
        remainder_count = {}
        
        for hour in hours:
            remainder = hour % 24
            remainder_count[remainder] = remainder_count.get(remainder, 0) + 1
        
        count = 0
        
        # Handle remainder 0 case (pairs with itself)
        if 0 in remainder_count:
            n = remainder_count[0]
            count += n * (n - 1) // 2  # Choose 2 from n elements
        
        # Handle remainder 12 case (pairs with itself since 12 + 12 = 24)
        if 12 in remainder_count:
            n = remainder_count[12]
            count += n * (n - 1) // 2
        
        # Handle other remainders (pair with their complement)
        for r in range(1, 12):
            if r in remainder_count and (24 - r) in remainder_count:
                count += remainder_count[r] * remainder_count[24 - r]
        
        return count