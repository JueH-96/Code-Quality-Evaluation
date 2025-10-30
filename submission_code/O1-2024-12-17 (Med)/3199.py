class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        from math import comb
        
        # Inclusion-exclusion approach:
        # Count the number of nonnegative integer solutions to x1 + x2 + x3 = n
        # subject to each xi <= limit.
        # 
        # Without any restriction, the count is C(n+2, 2).
        # We then exclude solutions where at least one x_i > limit, add back
        # where at least two exceed limit, etc.
        
        def valid_comb(a, b):
            # Returns comb(a, b) if a >= b >= 0, otherwise 0
            if a < b or b < 0 or a < 0:
                return 0
            return comb(a, b)
        
        total = 0
        # k is the number of kids that exceed (limit)
        # We apply inclusion-exclusion up to the 3 kids
        for k in range(4):  # k = 0, 1, 2, 3
            # If k kids each get at least (limit+1) candies,
            # we subtract k*(limit+1) from the total candies
            # and count the ways to distribute the remainder among 3 kids.
            remainder = n - k * (limit + 1)
            count_ways = valid_comb(remainder + 2, 2)
            
            # Add or subtract this count depending on parity of k
            if k % 2 == 0:  # even k => add
                total += comb(3, k) * count_ways
            else:           # odd k => subtract
                total -= comb(3, k) * count_ways
        
        return total