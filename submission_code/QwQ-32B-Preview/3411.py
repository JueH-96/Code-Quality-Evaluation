import math
from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def cumulative_popcount(n):
            if n == 0:
                return 0
            highest = math.floor(math.log2(n))
            return (highest * (1 << (highest - 1))) + cumulative_popcount(n - (1 << highest)) + (n - (1 << highest) + 1)
        
        def sum_exponents(m):
            # Find the maximum i such that cumulative_popcount(i) <= m
            low = 0
            high = 1 << 60  # Arbitrary large number
            while low < high:
                mid = (low + high + 1) // 2
                if cumulative_popcount(mid) > m:
                    high = mid - 1
                else:
                    low = mid
            max_i = low
            # Sum over j from 0 to 60: j * number of i's with j-th bit set up to max_i
            total = 0
            for j in range(60):
                count_j = (max_i + 1) >> (j + 1)
                if (max_i + 1) & (1 << j):
                    count_j += (max_i + 1) & ((1 << j) - 1)
                total += j * count_j
            return total
        
        def compute_product(from_i, to_i, mod_i):
            if from_i > to_i:
                return 1
            sum_to_i = sum_exponents(to_i)
            sum_from_i_minus_1 = sum_exponents(from_i - 1) if from_i >= 1 else 0
            exponent = sum_to_i - sum_from_i_minus_1
            return pow(2, exponent, mod_i)
        
        answer = []
        for query in queries:
            from_i, to_i, mod_i = query
            result = compute_product(from_i, to_i, mod_i)
            answer.append(result)
        return answer