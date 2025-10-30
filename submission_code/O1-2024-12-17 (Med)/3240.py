class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        """
        We want the maximum integer N such that the sum of 'prices' for all numbers
        from 1 to N is <= k.
        
        The 'price' of a single number num is the count of set bits in positions
        that are multiples of x (1-indexed from right to left).
        
        For example, if x=2, then only bits #2, #4, #6, ... (right to left, 1-indexed)
        are considered in the price of each number.
        
        We'll use a binary search over N and a function to compute sumOfPrices(N, x).
        
        sumOfPrices(N, x) = sum_{p in multiples_of_x} countBitsInPosition(N, p),
        where countBitsInPosition(N, p) = how many integers from 1..N have the p-th bit
        (1-indexed from the right) set = 1.
        
        We'll only need to check p up to around 60 or so, because for N up to ~10^18,
        bit positions above 60 won't matter (2^60 ~ 1.15e18).
        """
        
        def countBitsPosition(N: int, p: int) -> int:
            """
            Count how many numbers in [1..N] have their p-th bit (1-indexed from right) set.
            We'll shift to 0-based indexing internally: p0 = p - 1.
            
            For 0-based bit p0, each block of size 2^(p0+1) has that bit set exactly
            2^p0 times (the second half of each block). Using the well-known pattern:
            
            countSetBitInPosition(N, p0) from 0..N = 
                (N+1) // blockSize * halfBlock + max(0, (N+1) % blockSize - halfBlock)
            
            Here blockSize = 2^(p0+1) and halfBlock = 2^p0.
            We then interpret that for numbers in [1..N], the effect of 0 is negligible,
            since 0 does not contribute any set bits. So the formula works as is.
            """
            p0 = p - 1
            if p0 < 0:
                return 0
            
            block_size = 1 << (p0 + 1)  # 2^(p0+1)
            half_block = 1 << p0       # 2^p0
            
            full_blocks = (N + 1) // block_size   # how many complete blocks in [0..N]
            remainder  = (N + 1)  %  block_size   # leftover after full blocks
            
            # Each full block contributes 'half_block' set bits for this position
            res = full_blocks * half_block
            
            # For the remainder, the bit is set if remainder > half_block
            # in that leftover portion
            if remainder > half_block:
                res += remainder - half_block
            
            return res
        
        def sumOfPrices(N: int, x: int) -> int:
            """
            Sum of prices from 1..N = sum over each position p that is a multiple of x
            of (how many numbers in [1..N] have bit p set).
            We'll check p up to ~60 to cover up to ~2^60.
            """
            total = 0
            # Check bit positions p = x, 2x, 3x, ... up to ~60
            # (since 2^60 ~ 1.15e18, beyond that won't matter for N <= 2^60)
            for p in range(x, 62, x):
                total += countBitsPosition(N, p)
            return total
        
        # We now do a binary search for the largest N such that sumOfPrices(N, x) <= k.
        # We choose an upper bound that is safely large (e.g. 2^60 or a bit more).
        left, right = 0, min((1 << 60), 2 * k * x + 200)  # some safety margin
        
        while left < right:
            mid = (left + right + 1) // 2
            if sumOfPrices(mid, x) <= k:
                left = mid
            else:
                right = mid - 1
        
        return left