class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        
        result = x
        for i in range(1, n):
            result |= (1 << (self.get_msb(result) + 1))
        
        return result

    def get_msb(self, n):
        msb = -1
        while n > 0:
            n = n >> 1
            msb += 1
        return msb