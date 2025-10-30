from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_set_bits_upto(x):
            if x == 0:
                return 0
            total = 0
            k = 0
            while (1 << k) <= x:
                base = 1 << k
                next_base = base << 1
                full_cycles = (x + 1) // next_base
                remainder = (x + 1) % next_base
                cnt = full_cycles * base
                cnt += max(0, remainder - base)
                total += cnt
                k += 1
            return total

        def sum_exponents_upto(x):
            if x == 0:
                return 0
            total = 0
            k = 0
            while (1 << k) <= x:
                base = 1 << k
                next_base = base << 1
                full_cycles = (x + 1) // next_base
                remainder = (x + 1) % next_base
                cnt = full_cycles * base
                cnt += max(0, remainder - base)
                total += cnt * k
                k += 1
            return total

        def find_x(M):
            low = 1
            high = 1
            while count_set_bits_upto(high) < M:
                high *= 2
            while low < high:
                mid = (low + high) // 2
                cnt = count_set_bits_upto(mid)
                if cnt < M:
                    low = mid + 1
                else:
                    high = mid
            return low

        def compute_S(M):
            if M == 0:
                return 0
            x = find_x(M)
            cnt_x_minus_1 = count_set_bits_upto(x - 1)
            t = M - cnt_x_minus_1
            sum_total = sum_exponents_upto(x - 1)
            original_x = x
            sum_partial = 0
            count = 0
            k = 0
            while original_x > 0 and count < t:
                if original_x & 1:
                    sum_partial += k
                    count += 1
                original_x >>= 1
                k += 1
            return sum_total + sum_partial

        answer = []
        for from_i, to_i, mod_i in queries:
            L = from_i + 1
            R = to_i + 1
            if L == 0:
                sum_exp = compute_S(R)
            else:
                sum_exp = compute_S(R) - compute_S(L - 1)
            if mod_i == 1:
                answer.append(0)
            else:
                answer.append(pow(2, sum_exp, mod_i))
        return answer