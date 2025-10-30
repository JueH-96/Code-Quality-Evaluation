class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        # We want to find the maximum n such that the sum of prices
        # of all numbers from 1 to n is <= k.
        #
        # price(num) = count of set bits in num's binary representation
        #              at positions that are multiples of x
        #              (1-indexed from the right).
        #
        # We'll use binary search over n. For each candidate n, we
        # compute sum_price(n, x), i.e. the sum of price(i) for i=1..n.
        # Then we compare sum_price(n, x) with k.
        #
        # sum_price(n, x) = sum_{p = x, 2x, 3x, ...} count_set_bit_in_position(n, p)
        #
        # Here, count_set_bit_in_position(n, p) is the number of i in [1..n]
        # whose p-th bit (1-indexed) is set.
        #
        # There is a known formula to count how many numbers in [1..n] have
        # a set bit at position p. Let p0 = p - 1 (0-indexed).
        # Then the bit repeats in blocks of size 2^p. 
        # The p-th bit is set for the second half of each block (i.e., 2^(p-1) numbers in each block).
        #
        # count_set_bit_in_position(n, p) =
        #     (n // 2^p) * 2^(p-1) + max(0, (n % 2^p) - 2^(p-1) + 1)
        #
        # We'll implement this count, then sum over p = x, 2x, 3x, ... up to log2(n)+1.
        #
        # We do a binary search for n in [0..some large upper bound],
        # e.g. up to 2^52 to be safe, since k <= 10^15.

        def count_bit_in_position(n: int, p: int) -> int:
            # Counts how many numbers from 1..n (inclusive)
            # have the p-th bit (1-indexed from the right) set.
            if n == 0:
                return 0
            block_size = 1 << p  # 2^p
            half_block = block_size >> 1  # 2^(p-1)
            full_blocks = n // block_size
            remainder = n % block_size
            count = full_blocks * half_block
            if remainder >= half_block:
                count += (remainder - half_block + 1)
            return count

        def sum_price(n: int, x: int) -> int:
            # Sums the prices for all nums in [1..n].
            # Only bits at positions multiple of x matter.
            if n == 0:
                return 0
            import math
            max_p = n.bit_length()  # ~ floor(log2(n)) + 1
            total = 0
            p = x
            while p <= max_p:
                total += count_bit_in_position(n, p)
                p += x
            return total

        # Binary search for the largest n such that sum_price(n, x) <= k
        left, right = 0, 1 << 52  # a sufficiently large upper bound

        while left < right:
            mid = (left + right + 1) // 2
            if sum_price(mid, x) <= k:
                left = mid
            else:
                right = mid - 1

        return left