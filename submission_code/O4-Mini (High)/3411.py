from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # We treat big_nums as 0-indexed.
        # big_nums is the concatenation of binary decompositions (powers of two)
        # of 1,2,3,..., sorted ascending per number.
        # Each element is 2^e, so the product over a range is 2^(sum of exponents).
        
        # Precompute small powers of two for bit manipulations
        MAXB = 61
        pow2 = [1 << i for i in range(MAXB)]
        
        # bits_count(n): total number of bits in binary representations of 1..n
        def bits_count(n: int) -> int:
            if n <= 0:
                return 0
            cnt = 0
            np1 = n + 1
            bl = n.bit_length()
            for k in range(bl):
                # cycle length = 2^(k+1)
                block = pow2[k+1]
                full_cycles = np1 >> (k+1)
                ones = full_cycles << k
                rem = np1 & (block - 1)
                extra = rem - pow2[k]
                if extra > 0:
                    ones += extra
                cnt += ones
            return cnt
        
        # sum_exponents(n): sum of bit‐positions (exponents) in 1..n
        def sum_exponents(n: int) -> int:
            if n <= 0:
                return 0
            s = 0
            np1 = n + 1
            bl = n.bit_length()
            for k in range(bl):
                block = pow2[k+1]
                full_cycles = np1 >> (k+1)
                ones = full_cycles << k
                rem = np1 & (block - 1)
                extra = rem - pow2[k]
                if extra > 0:
                    ones += extra
                s += ones * k
            return s
        
        # Find an upper bound hi so that bits_count(hi) > maxR
        maxR = 0
        for _, r, _ in queries:
            if r > maxR:
                maxR = r
        hi = 1
        # double until we cover all positions up to maxR
        while bits_count(hi) <= maxR:
            hi <<= 1
        
        # Given a 0-based position pos in big_nums, find the integer x
        # such that max index for numbers 1..x-1 is < pos <= max index for 1..x
        def find_num(pos: int) -> int:
            low, high = 1, hi
            # we want smallest x with bits_count(x) > pos
            while low < high:
                mid = (low + high) // 2
                if bits_count(mid) > pos:
                    high = mid
                else:
                    low = mid + 1
            return low
        
        # Get the sorted list of exponents for number n
        def get_exp_list(n: int) -> List[int]:
            bl = n.bit_length()
            lst = []
            for k in range(bl):
                if (n >> k) & 1:
                    lst.append(k)
            return lst
        
        ans = []
        for L, R, mod in queries:
            # find which numbers contain the L-th and R-th big_nums elements
            a = find_num(L)
            b = find_num(R)
            # offset inside that number's bit-list (0-based)
            off_a = L - bits_count(a - 1)
            off_b = R - bits_count(b - 1)
            
            if a == b:
                # both L and R within the same number
                e_list = get_exp_list(a)
                S = sum(e_list[off_a:off_b + 1])
            else:
                # start partial (from off_a to end of a)
                e_list_a = get_exp_list(a)
                S_start = sum(e_list_a[off_a:])
                # end partial (from start of b to off_b)
                e_list_b = get_exp_list(b)
                S_end = sum(e_list_b[:off_b + 1])
                # full middle numbers a+1..b-1
                if b > a + 1:
                    S_mid = sum_exponents(b - 1) - sum_exponents(a)
                else:
                    S_mid = 0
                S = S_start + S_mid + S_end
            
            # product is 2^S mod mod
            ans.append(pow(2, S, mod))
        
        return ans