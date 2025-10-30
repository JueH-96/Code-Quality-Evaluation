from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # ------------------------------------------------------------------
        # Helpers dealing with the statistical properties of the sequence
        # ------------------------------------------------------------------
        #
        # big_nums is produced by writing, for every positive integer n,
        # the powers of two that occur in the binary representation of n,
        # ordered from the lowest bit to the highest one.
        #
        # Each element of big_nums is therefore exactly 2^k for some k,
        # and the product over any interval can be written as
        #        2 ^ (sum_of_k_on_interval)
        #
        # Hence we only need the prefix sums of the exponents k, not the
        # numbers themselves.
        #
        # For numbers 1 .. n
        #   f(n) = total amount of produced elements      (pop-count sum)
        #   g(n) = sum of all produced exponents k        (needed power)
        #
        # Both f and g can be obtained by scanning every bit position
        # (there are at most 60 for n ≤ 10^15).
        #
        # With them, for an arbitrary position P (number of produced
        # elements), we can:
        #   • binary-search the largest integer n with f(n) ≤ P
        #   • add g(n)                                     (complete part)
        #   • add the first (P – f(n)) exponents of n+1    (partial part)
        #
        # Prefix(P) finally returns the sum of exponents of the first
        # P elements (indices 0 … P-1).
        #
        # ------------------------------------------------------------------
        import functools

        # -------- total number of ones (produced elements) in 1..n --------
        def total_ones_upto(n: int) -> int:
            """f(n) – total number of produced elements for 1..n"""
            if n <= 0:
                return 0
            cnt = 0
            k = 0
            while (1 << k) <= n:
                cycle_len = 1 << (k + 1)
                full_cycles = (n + 1) // cycle_len
                cnt += full_cycles * (1 << k)
                remainder = (n + 1) % cycle_len
                extra = remainder - (1 << k)
                if extra > 0:
                    cnt += extra
                k += 1
            return cnt

        # -------- sum of exponents of all produced elements in 1..n -------
        def sum_exponents_upto(n: int) -> int:
            """g(n) – sum of exponents for 1..n"""
            if n <= 0:
                return 0
            total = 0
            k = 0
            while (1 << k) <= n:
                cycle_len = 1 << (k + 1)
                full_cycles = (n + 1) // cycle_len
                ones = full_cycles * (1 << k)
                remainder = (n + 1) % cycle_len
                extra = remainder - (1 << k)
                if extra > 0:
                    ones += extra
                total += ones * k
                k += 1
            return total

        # ------------------- binary search helper -------------------------
        def find_last_n_with_enough_elements(P: int) -> int:
            """
            Return the largest integer n such that total_ones_upto(n) ≤ P.
            P is a non-negative integer (number of produced elements).
            """
            low, high = 0, P  # f(high) ≥ high ≥ P  ⇒ safe upper bound
            while low < high:
                mid = (low + high + 1) // 2
                if total_ones_upto(mid) <= P:
                    low = mid
                else:
                    high = mid - 1
            return low  # low == high

        # ------------------- prefix sum of exponents ----------------------
        def prefix_sum_exponents(P: int) -> int:
            """
            Sum of exponents of the first P elements of big_nums
            (indices 0 … P-1).   P ≥ 0
            """
            if P == 0:
                return 0
            n = find_last_n_with_enough_elements(P)
            full_cnt = total_ones_upto(n)
            res = sum_exponents_upto(n)

            # Need the first (P – full_cnt) exponents of number n+1
            remaining = P - full_cnt
            if remaining:
                num = n + 1
                bit = 0
                while remaining:
                    if num & (1 << bit):
                        res += bit
                        remaining -= 1
                    bit += 1
            return res

        # ------------------------------------------------------------------
        # Main part – answer every query
        # ------------------------------------------------------------------
        answers = []
        for left, right, mod in queries:
            if mod == 1:                      # every number mod 1 is 0
                answers.append(0)
                continue

            exp_right = prefix_sum_exponents(right + 1)   # inclusive
            exp_left  = prefix_sum_exponents(left)        # exclusive
            exponent  = exp_right - exp_left              # non-negative

            answers.append(pow(2, exponent, mod))

        return answers