import math

class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        # Calculate the total number of ways without constraints
        total = math.comb(n + 2, 2)
        
        # Calculate the number of ways where one child gets more than limit candies
        if n - (limit + 1) >= 0:
            subtraction = 3 * math.comb(n - (limit + 1) + 2, 2)
        else:
            subtraction = 0
        
        # The valid number of ways is the total minus the invalid distributions
        return total - subtraction