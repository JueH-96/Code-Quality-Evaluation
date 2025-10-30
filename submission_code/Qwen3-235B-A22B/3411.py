from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_bit_set_upto(x, k):
            if x < 0 or k < 0:
                return 0
            block_size = (1 << (k + 1))
            half_block = (1 << k)
            full_cycles = (x + 1) // block_size
            count = full_cycles * half_block
            remainder = (x + 1) % block_size
            count += max(0, remainder - half_block)
            return count

        def compute_C(x):
            if x < 1:
                return 0
            res = 0
            for k in range(60):
                res += count_bit_set_upto(x, k)
            return res

        def compute_S(x):
            if x < 1:
                return 0
            res = 0
            for k in range(60):
                cnt = count_bit_set_upto(x, k)
                res += cnt * k
            return res

        def find_x(pos):
            low = 0
            high = 1 << 60
            while low < high:
                mid = (low + high) // 2
                c_mid = compute_C(mid)
                if c_mid <= pos:
                    low = mid + 1
                else:
                    high = mid
            return low

        def get_exponents(x):
            if x == 0:
                return []
            exponents = []
            k = 0
            while (1 << k) <= x:
                if x & (1 << k):
                    exponents.append(k)
                k += 1
            return exponents

        def compute_S_total(pos):
            if pos < 0:
                return 0
            x = find_x(pos)
            x_prev = x - 1
            c_prev = compute_C(x_prev)
            remaining = pos - c_prev
            s_prev = compute_S(x_prev)
            exponents_x = get_exponents(x)
            sum_partial = sum(exponents_x[:remaining + 1]) if remaining + 1 > 0 else 0
            return s_prev + sum_partial

        res = []
        for a, b, mod in queries:
            S_high = compute_S_total(b)
            S_low = compute_S_total(a - 1)
            S = S_high - S_low
            if mod == 1:
                ans = 0
            else:
                ans = pow(2, S, mod)
            res.append(ans)
        return res