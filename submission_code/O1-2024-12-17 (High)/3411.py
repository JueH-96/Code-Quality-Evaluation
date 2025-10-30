from functools import lru_cache
from typing import List

class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        # We need to answer queries of the form:
        #   product( big_nums[from_i] .. big_nums[to_i] ) % mod_i
        # where big_nums is formed by concatenating the "powerful arrays" for 1,2,3,...
        #
        # A "powerful array" for an integer x is the minimal sorted list of powers of two summing to x;
        # effectively, these are the bits set in x (in ascending order of powers).
        #
        # Example: x=11 (binary 1011) => [1,2,8].  Sorted ascending => [1,2,8].
        #
        # The sequence big_nums begins:
        #   i=1 => popcount(1)=1 => [1]
        #   i=2 => popcount(2)=1 => [2]
        #   i=3 => popcount(3)=2 => [1,2]
        #   i=4 => popcount(4)=1 => [4]
        #   i=5 => popcount(5)=2 => [1,4]
        #   ...
        # which yields big_nums = [1, 2, 1, 2, 4, 1, 4, 2, 4, 1, 2, 4, 8, ...].
        #
        # Indices in big_nums are 0-based.  We can have queries up to very large indices (to 10^15).
        # We obviously cannot build big_nums explicitly up to those indices.  Instead, we:
        #
        # 1) Note that the length of the powerful array for i is popcount(i).
        #    So if we define S(n) = sum_{k=1..n} popcount(k),
        #    then big_nums[ S(n-1) .. S(n)-1 ] correspond to the powers-of-two for integer n.
        #
        # 2) To find which integer covers a particular global index X in big_nums, we find n
        #    such that S(n-1) <= X < S(n).  Then offset = X - S(n-1) is which element in n's
        #    powerful array (0-based).
        #
        # 3) The product of a single integer i's powerful array is 2^( sum_of_set_bit_positions(i) ).
        #    Because if i has bits at positions b1, b2, ... (where the least-significant bit is position 0),
        #    then the powerful array is [2^b1, 2^b2, ...], and their product is 2^(b1 + b2 + ...).
        #
        # 4) For a query over a range [from_i..to_i], we can break that into:
        #      - a partial tail of i1 (the integer covering from_i),
        #      - all full integers between i1+1 and i2-1,
        #      - a partial head of i2 (the integer covering to_i).
        #
        # 5) Summation approach: 
        #    If f(i) = sum_of_set_bit_positions(i),
        #    then the product from i=a..b is 2^( sum_{k=a..b} f(k) ), if we take the entire powerful arrays.
        #    For partial pieces, we only sum the positions of the selected bits.
        #
        # 6) We implement:
        #    - sumPopcount(n): returns S(n) = sum_{k=1..n} popcount(k).
        #    - sumBitPositions(n): returns sum_{k=1..n} f(k).
        #    - findNumberOffset(x): finds i and offset so that S(i-1) <= x < S(i).
        #    - partialSumBitPositions(i, start, end): sums the bit-positions of the start..end set bits of i.
        #
        # Then the query product modulo M is pow(2, exponent, M), where
        # exponent = partial_tail_of_i1 + sum_{k=i1+1..i2-1} f(k) + partial_head_of_i2.
        #
        # Complexity is kept manageable by O(log n) computations and by using fast exponentiation.

        # ---------------------------------------------------------
        # 1) Precompute functions for popcount-sums and bit-position-sums, with caching.
        # ---------------------------------------------------------

        @lru_cache(None)
        def sumPopcount(n: int) -> int:
            """
            Returns sum_{k=1..n} popcount(k).
            This is the index in big_nums of the first element after all integers up to n.
            """
            if n <= 0:
                return 0
            # Count how many times each bit is set from 1..n
            # The bit b toggles in blocks of size 2^(b+1), being set for 2^b of each block.
            # We'll do this in O(log n).
            total = 0
            limit = n.bit_length()
            for b in range(limit):
                block_size = 1 << (b+1)  # 2^(b+1)
                full_blocks = n // block_size
                remainder = n % block_size
                # bit b is set full_blocks * (2^b) times in the full blocks
                # plus the leftover in the partial block:
                total += full_blocks * (1 << b)
                # partial block:
                left_in_block = max(0, remainder - (1 << b) + 1)
                if left_in_block > 0:
                    total += left_in_block
            return total

        @lru_cache(None)
        def sumBitPositions(n: int) -> int:
            """
            Returns sum_{k=1..n} of (sum of set-bit-positions in k).
            If k in binary has bits set at positions b1, b2, ..., then f(k) = b1 + b2 + ...
            We sum f(k) for k=1..n.
            """
            if n <= 0:
                return 0
            # We can compute it by summing over bits b: b * (count of numbers 1..n with bit b set).
            # Then for each bit b, we multiply b * countBitSet(b, n).
            # Same O(log n) approach:
            total = 0
            limit = n.bit_length()
            for b in range(limit):
                block_size = 1 << (b+1)
                full_blocks = n // block_size
                remainder = n % block_size
                count_set_b = full_blocks * (1 << b)
                leftover = max(0, remainder - (1 << b) + 1)
                count_set_b += leftover if leftover > 0 else 0
                total += b * count_set_b
            return total

        def findNumberOffset(idx: int):
            """
            Given a 0-based index idx into big_nums, find the integer i and offset in i's powerful array
            such that big_nums[idx] is the 'offset'-th element of i's powerful array.
            
            i.e. we want i such that sumPopcount(i-1) <= idx < sumPopcount(i).
            Then offset = idx - sumPopcount(i-1).
            """
            if idx == 0:
                # The very first element is i=1, offset=0
                return (1, 0)
            # Exponential search to find an upper bound
            hi = 1
            while sumPopcount(hi) <= idx:
                hi <<= 1
            # Now binary search in [hi//2+1..hi]
            lo = hi >> 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if sumPopcount(mid) > idx:
                    hi = mid
                else:
                    lo = mid + 1
            i = lo      # smallest i such that sumPopcount(i) > idx
            # offset is how many steps into i's segment
            offset = idx - sumPopcount(i-1)
            return (i, offset)

        def partialSumBitPositions(x: int, start_bit_index: int, end_bit_index: int) -> int:
            """
            In the integer x, gather the positions of set bits in ascending order.
            Return the sum of those bit positions from index start_bit_index..end_bit_index (inclusive).
            
            For example, if x=13 (binary 1101), bit_positions(x) = [0,2,3].
            partialSumBitPositions(13, 0, 1) = positions[0]+positions[1] = 0+2=2.
            """
            if start_bit_index > end_bit_index:
                return 0
            # Collect set bit positions
            positions = []
            tmp = x
            bit_pos = 0
            while tmp > 0:
                if (tmp & 1) != 0:
                    positions.append(bit_pos)
                tmp >>= 1
                bit_pos += 1
            # prefix sum
            psum = [0]*(len(positions)+1)
            for i in range(len(positions)):
                psum[i+1] = psum[i] + positions[i]
            # clamp indexes
            if start_bit_index < 0:
                start_bit_index = 0
            if end_bit_index >= len(positions):
                end_bit_index = len(positions) - 1
            if start_bit_index > end_bit_index or start_bit_index < 0:
                return 0
            return psum[end_bit_index+1] - psum[start_bit_index]

        answers = []
        for (frm, to, mod) in queries:
            if mod == 1:
                # everything mod 1 is zero
                answers.append(0)
                continue
            if frm > to:
                # No elements - product = 1 in normal math, so mod => 1 % mod
                answers.append(1 % mod)
                continue

            # Find which integer covers frm, and offset
            i1, off1 = findNumberOffset(frm)
            # Find which integer covers to, and offset
            i2, off2 = findNumberOffset(to)

            if i1 == i2:
                # Entire range is within the same integer's powerful array
                exponent = partialSumBitPositions(i1, off1, off2)
            else:
                # partial tail of i1
                tail_exp = partialSumBitPositions(i1, off1, (i1.bit_count() - 1))
                # partial head of i2
                head_exp = partialSumBitPositions(i2, 0, off2)
                # full integers from i1+1.. i2-1
                mid_exp = 0
                if i2 > i1 + 1:
                    mid_exp = sumBitPositions(i2 - 1) - sumBitPositions(i1)
                exponent = tail_exp + head_exp + mid_exp

            # Compute 2^exponent mod M
            ans = pow(2, exponent, mod)
            answers.append(ans)

        return answers