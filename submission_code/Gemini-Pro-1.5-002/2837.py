class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for ops in range(1, 33):
            diff = num1 - ops * num2
            if diff < ops:
                continue
            
            bits = 0
            temp = diff
            while temp > 0:
                bits += temp & 1
                temp >>= 1
            
            if bits <= ops and ops <= diff:
                return ops
        return -1