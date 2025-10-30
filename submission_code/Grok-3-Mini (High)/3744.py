from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        def prefix_sum(p: int) -> int:
            if p <= 0:
                return 0
            log2_p_floor = p.bit_length() - 1
            m = (log2_p_floor // 2) + 1
            if m == 1:
                sum_prev = 0
            else:
                exp = m - 1
                coeff = 3 * m - 4
                sum_prev = ((coeff * (4 ** exp) + 1) // 3)
            a_m = 4 ** (m - 1)
            sum_partial = m * (p - a_m + 1)
            return sum_prev + sum_partial
        
        total = 0
        for l, r in queries:
            sum_d = prefix_sum(r) - prefix_sum(l - 1)
            min_ops = (sum_d + 1) // 2
            total += min_ops
        return total