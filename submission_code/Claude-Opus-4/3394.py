class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Start with x as the first element
        result = x
        
        # We need to add (n-1) to get the nth element
        # But we can only use bits that are 0 in x
        remaining = n - 1
        
        # Find positions where x has 0 bits
        bit_position = 0
        
        while remaining > 0:
            # Check if current bit position in x is 0
            if (x >> bit_position) & 1 == 0:
                # We can use this bit position
                if remaining & 1:
                    result |= (1 << bit_position)
                remaining >>= 1
            bit_position += 1
        
        return result