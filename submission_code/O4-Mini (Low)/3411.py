from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # Count total 1-bits in binary representations of 1..n
        def total_bits(n: int) -> int:
            # P(n) = sum_{k>=0} count of ones at bit k from 1..n
            cnt = 0
            k = 0
            while (1 << k) <= n:
                block = 1 << (k + 1)
                full = (n + 1) // block
                rem = (n + 1) % block
                ones = full * (1 << k) + max(0, rem - (1 << k))
                cnt += ones
                k += 1
            return cnt

        # Sum of exponents (bit positions) for all 1-bits in 1..n
        def total_exponents(n: int) -> int:
            s = 0
            k = 0
            while (1 << k) <= n:
                block = 1 << (k + 1)
                full = (n + 1) // block
                rem = (n + 1) % block
                ones = full * (1 << k) + max(0, rem - (1 << k))
                s += k * ones
                k += 1
            return s

        # Sum of exponents for the first t entries in the big_nums stream
        # big_nums is the concatenation, for i=1..inf, of the powers-of-two (as exponents k)
        def sum_prefix(t: int) -> int:
            # t entries means positions 0..t-1 inclusive, so t must be >= 0
            if t <= 0:
                return 0
            # binary search for maximum n so that total_bits(n) < t
            lo, hi = 0, t  # n in [0..t]
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if total_bits(mid) < t:
                    lo = mid
                else:
                    hi = mid - 1
            n = lo
            used = total_bits(n)
            rem = t - used  # how many bits to take from number (n+1)
            s = total_exponents(n)
            if rem > 0:
                # take the first 'rem' set bits of (n+1), in ascending k order
                x = n + 1
                k = 0
                while rem > 0 and x > 0:
                    if x & 1:
                        s += k
                        rem -= 1
                    x >>= 1
                    k += 1
            return s

        ans = []
        for l, r, m in queries:
            # indices are 0-based, sum entries l..r inclusive => prefix(r+1)-prefix(l)
            tot_exp = sum_prefix(r + 1) - sum_prefix(l)
            # product is 2^tot_exp mod m
            ans.append(pow(2, tot_exp, m))
        return ans