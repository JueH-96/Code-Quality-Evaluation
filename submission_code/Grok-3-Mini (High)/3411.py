import math
from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def sum_popcount(n):
            total = 0
            for k in range(60):
                p = 1 << k
                q = 1 << (k + 1)
                a = n // q
                b = n % q
                count_k = a * p
                if b >= p:
                    count_k += b - p + 1
                total += count_k
            return total
        
        def bit_set_count(n, k):
            p = 1 << k
            q = 1 << (k + 1)
            a = n // q
            b = n % q
            count = a * p
            if b >= p:
                count += b - p + 1
            return count
        
        def sum_exponents_up_to_m(m):
            total = 0
            for k in range(60):
                num_set = bit_set_count(m, k)
                total += k * num_set
            return total
        
        def get_exponents(num):
            exps = []
            val = num
            while val:
                lowest_bit = val & -val
                exp = (lowest_bit).bit_length() - 1
                exps.append(exp)
                val -= lowest_bit
            return exps  # sorted in ascending order
        
        def cumulative_sum_exp(pos):
            if pos < 0:
                return 0
            # Binary search for the smallest i such that sum_popcount(i) > pos
            left = 0
            right = 1 << 60  # 2**60
            while left < right:
                mid = (left + right) // 2
                if sum_popcount(mid) > pos:
                    right = mid
                else:
                    left = mid + 1
            i = left  # i is the number where pos belongs
            cum_len_prev = sum_popcount(i - 1) if i > 0 else 0  # cumulative length up to i-1
            offset = pos - cum_len_prev  # 0-based index within i's block
            sum_up_to_prev = sum_exponents_up_to_m(i - 1) if i > 0 else 0
            exps_i = get_exponents(i)
            sum_partial = sum(exps_i[: offset + 1])  # sum of first offset+1 exponents
            total_sum = sum_up_to_prev + sum_partial
            return total_sum
        
        # Process each query
        answer = []
        for query in queries:
            L, R, mod_val = query
            sum_exp = cumulative_sum_exp(R) - cumulative_sum_exp(L - 1)
            product = pow(2, sum_exp, mod_val)
            answer.append(product)
        return answer