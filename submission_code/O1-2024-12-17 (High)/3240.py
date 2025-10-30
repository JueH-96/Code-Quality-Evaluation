class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        
        # Counts how many integers in [1..N] have the j-th bit (1-based from right) set.
        def count_set_bits_in_position(N: int, j: int) -> int:
            #  j-th bit has a repeating pattern of length 2^j
            #  in which 2^(j-1) consecutive numbers have that bit = 1, then 2^(j-1) = 0, etc.
            #  We use a known formula to count how many numbers from 0..N have bit j set,
            #  then note that 0 doesn't contribute (bit is unset), so [1..N] is the same count.
            p = 1 << j         # Period = 2^j
            half = p >> 1      # 2^(j-1)
            block_count = (N + 1) // p
            remainder = (N + 1) % p
            return block_count * half + max(0, remainder - half)
        
        # Computes sum of prices for all numbers 1..N.
        # The price of a number is the count of bits set in positions that are multiples of x.
        def total_price(N: int, x: int) -> int:
            total = 0
            j = x
            # Only go up to where 2^(j-1) <= N (no bits set beyond that for numbers up to N)
            while j <= 60 and (1 << (j - 1)) <= N:
                total += count_set_bits_in_position(N, j)
                j += x
            return total
        
        # We do a binary search for the largest N such that total_price(N,x) <= k.
        low, high = 0, 1 << 60
        answer = 0
        
        while low <= high:
            mid = (low + high) // 2
            if total_price(mid, x) <= k:
                answer = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return answer