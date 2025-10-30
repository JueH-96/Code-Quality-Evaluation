class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def comb(n, r):
            if r < 0 or r > n:
                return 0
            if r == 0 or r == n:
                return 1
            
            # Calculate C(n, r) = n! / (r! * (n-r)!)
            result = 1
            for i in range(min(r, n - r)):
                result = result * (n - i) // (i + 1)
            return result
        
        # Total ways without upper bound constraint
        # We need non-negative solutions to x1 + x2 + x3 = n
        total = comb(n + 2, 2)
        
        # Subtract ways where at least one child gets more than limit candies
        # If child i gets at least (limit + 1) candies, we distribute remaining
        # (n - limit - 1) candies among 3 children
        if n > limit:
            total -= 3 * comb(n - limit - 1 + 2, 2)
        
        # Add back ways where at least two children get more than limit candies
        if n > 2 * limit:
            total += 3 * comb(n - 2 * (limit + 1) + 2, 2)
        
        # Subtract ways where all three children get more than limit candies
        if n > 3 * limit:
            total -= comb(n - 3 * (limit + 1) + 2, 2)
        
        return total