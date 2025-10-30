import math

class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        total_ways = 0
        for i in range(0, 4):  # since there are 3 children, i from 0 to 3
            sign = (-1) ** i
            comb_k_i = math.comb(3, i)
            # Calculate the adjusted n for the combination
            adjusted_n = n - i * (limit + 1) + 2
            # If adjusted_n < 2, combination is 0
            if adjusted_n < 2:
                comb_n = 0
            else:
                comb_n = math.comb(adjusted_n, 2)
            term = sign * comb_k_i * comb_n
            total_ways += term
        return total_ways