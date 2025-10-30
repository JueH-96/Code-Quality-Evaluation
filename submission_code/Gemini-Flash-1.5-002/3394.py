class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        
        if n > 30 :
            return x + (1 << 30) -1

        result = x
        for i in range(n - 1):
            bit = 0
            while (result & (1 << bit)) == 0:
                bit += 1
            result |= (1 << bit)

        return result