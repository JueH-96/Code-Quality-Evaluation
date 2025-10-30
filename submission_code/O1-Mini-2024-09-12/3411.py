from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def get_sum_set_bits(N):
            total = 0
            for k in range(60):
                power = 1 << (k + 1)
                full_cycles = N // power
                bits = full_cycles * (1 << k)
                remainder = N % power
                bits += max(0, remainder - (1 << k) + 1)
                total += bits
            return total

        def get_sum_of_k_set_bits_up_to(N):
            total = 0
            for k in range(60):
                power = 1 << (k + 1)
                full_cycles = (N + 1) // power
                bits = full_cycles * (1 << k)
                remainder = (N + 1) % power
                bits += max(0, remainder - (1 << k))
                total += bits * k
            return total

        def get_sum_to(p):
            if p == 0:
                return 0
            # Binary search to find N
            left, right = 1, 1
            while get_sum_set_bits(right) < p:
                right <<= 1
            while left < right:
                mid = (left + right) // 2
                s = get_sum_set_bits(mid)
                if s >= p:
                    right = mid
                else:
                    left = mid + 1
            N = left
            sum1 = get_sum_of_k_set_bits_up_to(N - 1)
            sum_s = get_sum_set_bits(N - 1)
            remaining = p - sum_s
            bits = []
            num = N
            for k in range(60):
                if num & (1 << k):
                    bits.append(k)
            bits.sort()
            sum_added = sum(bits[:remaining]) if remaining > 0 else 0
            return sum1 + sum_added

        answers = []
        for from_i, to_i, mod_i in queries:
            sum_to_i = get_sum_to(to_i)
            sum_from = get_sum_to(from_i - 1)
            sum_p = sum_to_i - sum_from
            if mod_i == 1:
                result = 0
            else:
                result = pow(2, sum_p, mod_i)
            answers.append(result)
        return answers