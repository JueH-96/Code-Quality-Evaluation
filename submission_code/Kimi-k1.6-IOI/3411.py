class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        def count_set_bits(n):
            if n == 0:
                return 0
            count = 0
            k = 0
            while True:
                if (1 << k) > n:
                    break
                mask = 1 << (k + 1)
                high = (n + 1) // mask
                remainder = (n + 1) % mask
                cnt = high * (1 << k)
                cnt += max(0, remainder - (1 << k))
                count += cnt
                k += 1
            return count
        
        def sum_exponents(n):
            total = 0
            k = 0
            while True:
                if (1 << k) > n:
                    break
                mask = 1 << (k + 1)
                high = (n + 1) // mask
                remainder = (n + 1) % mask
                cnt = high * (1 << k)
                cnt += max(0, remainder - (1 << k))
                total += k * cnt
                k += 1
            return total
        
        def get_exponents(m):
            exponents = []
            pos = 0
            while m > 0:
                if m & 1:
                    exponents.append(pos)
                m >>= 1
                pos += 1
            return exponents
        
        def get_sum(pos):
            if pos == -1:
                return 0
            k = pos + 1
            low, high = 1, 2 * 10**15
            i_max = 0
            while low <= high:
                mid = (low + high) // 2
                s = count_set_bits(mid)
                if s <= k:
                    i_max = mid
                    low = mid + 1
                else:
                    high = mid - 1
            s_i_max = count_set_bits(i_max)
            rem = k - s_i_max
            sum_exp = sum_exponents(i_max)
            m = i_max + 1
            exponents_m = get_exponents(m)
            take = min(rem, len(exponents_m))
            sum_exp += sum(exponents_m[:take])
            return sum_exp
        
        result = []
        for L, R, mod in queries:
            sum_r = get_sum(R)
            sum_l = get_sum(L - 1) if L > 0 else 0
            exponent = sum_r - sum_l
            res = pow(2, exponent, mod)
            result.append(res)
        
        return result