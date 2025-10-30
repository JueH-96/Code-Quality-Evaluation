class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_set_bits(N):
            if N <= 0:
                return 0
            total = 0
            k = 0
            while True:
                shift_k = 1 << k
                if shift_k > N:
                    break
                period = 1 << (k + 1)
                full_cycles = (N + 1) // period
                remainder = (N + 1) % period
                count = full_cycles * shift_k
                if remainder > shift_k:
                    count += remainder - shift_k
                total += count
                k += 1
            return total

        def exponent_sum(N):
            if N <= 0:
                return 0
            total = 0
            k = 0
            while True:
                shift_k = 1 << k
                if shift_k > N:
                    break
                period = 1 << (k + 1)
                full_cycles = (N + 1) // period
                remainder = (N + 1) % period
                count = full_cycles * shift_k
                if remainder > shift_k:
                    count += remainder - shift_k
                total += k * count
                k += 1
            return total

        from functools import lru_cache
        @lru_cache(maxsize=None)
        def f(n):
            if n == 0:
                return 0
            low, high = 0, 10**18
            while low < high:
                mid = (low + high) // 2
                if count_set_bits(mid) < n:
                    low = mid + 1
                else:
                    high = mid
            M = low
            T_prev = count_set_bits(M - 1)
            rem = n - T_prev
            bits = []
            tmp = M
            exp_val = 0
            while tmp:
                if tmp & 1:
                    bits.append(exp_val)
                tmp //= 2
                exp_val += 1
            s = sum(bits[:rem])
            return exponent_sum(M - 1) + s

        res = []
        for from_i, to_i, mod_i in queries:
            E = f(to_i + 1) - f(from_i)
            res.append(pow(2, E, mod_i))
        return res