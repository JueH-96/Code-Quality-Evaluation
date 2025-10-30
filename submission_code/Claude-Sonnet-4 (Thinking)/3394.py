class Solution:
    def minEnd(self, n: int, x: int) -> int:
        result = x
        remaining = n - 1  # We need to encode (n-1) in the free bits
        bit_pos = 0
        
        while remaining > 0:
            # Check if current bit position is free (x has 0 there)
            if (x >> bit_pos) & 1 == 0:
                # If current bit of remaining is 1, set it in result
                if remaining & 1:
                    result |= (1 << bit_pos)
                remaining >>= 1
            bit_pos += 1
        
        return result