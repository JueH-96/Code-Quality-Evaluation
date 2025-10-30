class Solution:
    def minEnd(self, n: int, x: int) -> int:
        result = x
        remaining = n - 1  # We need to encode n-1 in the free bits
        
        bit_position = 0
        while remaining > 0:
            # Skip bits that are already set in x
            while (x >> bit_position) & 1:
                bit_position += 1
            
            # Set this bit in result if needed
            if remaining & 1:
                result |= (1 << bit_position)
            
            remaining >>= 1
            bit_position += 1
        
        return result