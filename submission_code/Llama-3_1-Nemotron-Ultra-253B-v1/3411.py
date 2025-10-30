from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def compute_S(x: int) -> int:
            if x == 0:
                return 0
            res = 0
            for i in range(60):
                res += count_i(x, i)
            return res

        def count_i(x: int, i: int) -> int:
            if x < 0:
                return 0
            cycle = 1 << (i + 1)
            cnt = (x + 1) // cycle * (1 << i)
            remainder = (x + 1) % cycle
            cnt += max(0, remainder - (1 << i))
            return cnt

        def find_x(n: int) -> int:
            if n == 0:
                return 1
            low = 1
            high = 1
            while compute_S(high) <= n:
                high *= 2
            while low <= high:
                mid = (low + high) // 2
                s = compute_S(mid - 1)
                if s <= n:
                    low = mid + 1
                else:
                    high = mid - 1
            return high

        def get_exponents(x: int) -> List[int]:
            exponents = []
            for i in range(60):
                if x & (1 << i):
                    exponents.append(i)
            return exponents

        def sum_exponents_middle(a: int, b: int) -> int:
            if a > b:
                return 0
            total = 0
            for i in range(60):
                cnt = count_i(b, i) - count_i(a - 1, i)
                total += i * cnt
            return total

        result = []
        for q in queries:
            L, R, mod = q
            x_start = find_x(L)
            s_start_minus_1 = compute_S(x_start - 1)
            pos_start = L - s_start_minus_1

            x_end = find_x(R)
            s_end_minus_1 = compute_S(x_end - 1)
            pos_end = R - s_end_minus_1

            if x_start == x_end:
                exponents = get_exponents(x_start)
                sum_exp = sum(exponents[pos_start:pos_end + 1])
                product = pow(2, sum_exp, mod)
            else:
                exponents_start = get_exponents(x_start)
                sum_start = sum(exponents_start[pos_start:])
                exponents_end = get_exponents(x_end)
                sum_end = sum(exponents_end[:pos_end + 1])
                sum_middle = sum_exponents_middle(x_start + 1, x_end - 1)
                total_exp = sum_start + sum_middle + sum_end
                product = pow(2, total_exp, mod)

            result.append(product % mod)

        return result