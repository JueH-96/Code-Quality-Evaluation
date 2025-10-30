class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        """
        We want the largest N such that sum_{n=1..N} price(n) <= k,
        where price(n) is the count of bits set in n's binary representation
        at 1-based indices i that satisfy (i % x == 0).

        Equivalently in 0-based bit indices, we count bits b where (b+1) % x == 0.

        We'll use a binary search for N. For each candidate N, we compute:
            sum_price(N) = sum_{n=1..N} price(n).
        Then compare it with k. If sum_price(N) <= k, we move the search up;
        otherwise we move it down.

        To compute sum_price(N) efficiently, note:
            sum_price(N) = sum_{b in B} countSetBitsInPosition(N, b),
        where B = { b >= 0 : (b+1) mod x == 0 } (up to ~60 bits for safety),
        and countSetBitsInPosition(N, b) is how many times bit b is set
        among the numbers 1..N.

        For a given bit b, the pattern for that bit (0 or 1) repeats
        every 2^(b+1) numbers, with exactly 2^b numbers being "1" in each block.
        The formula for countSetBitsInPosition(N, b) is:
            full_blocks = N // (2^(b+1))
            leftover    = N  %  (2^(b+1))
            count = full_blocks * 2^b + max(0, leftover - 2^b + 1)
        """

        # Precompute the bit positions we need to check:
        # (b+1) % x == 0 => b % x == x-1 => b = x-1, x-1 + x, ...
        # We'll go up to 60 bits (which covers up to 2^60 ~ 1.15e18).
        relevant_bits = []
        for b in range(60):  # 60 is enough to cover up to ~10^18 range
            if (b + 1) % x == 0:
                relevant_bits.append(b)

        def count_set_bits_in_position(N: int, b: int) -> int:
            """Count how many times bit b is set in [1..N]."""
            if N <= 0:
                return 0
            block_size = 1 << (b + 1)      # 2^(b+1)
            half_block = 1 << b           # 2^b

            full_blocks = N // block_size
            leftover = N % block_size

            count = full_blocks * half_block
            # In the leftover part, if leftover >= half_block,
            # then that bit is set leftover - half_block + 1 times.
            if leftover >= half_block:
                count += leftover - half_block + 1
            return count

        def total_price(N: int) -> int:
            """Sum of prices of all numbers from 1..N."""
            if N <= 0:
                return 0
            total = 0
            for b in relevant_bits:
                total += count_set_bits_in_position(N, b)
            return total

        # Binary search for the maximum N such that total_price(N) <= k
        left, right = 0, 1 << 60  # Search up to 2^60
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if total_price(mid) <= k:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1

        return ans