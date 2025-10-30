class Solution:
    def minEnd(self, n: int, x: int) -> int:
        result = x
        bit_position = 0
        
        while n > 0:
            if (x >> bit_position) & 1 == 0:
                result |= (1 << bit_position)
                n -= 1
            bit_position += 1
        
        return result