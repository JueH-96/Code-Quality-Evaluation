class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num2 >= 0:
            if num1 <= num2:
                return -1
        
        for k in range(1, 61):
            rem = num1 - k * num2
            if rem <= 0:
                continue
            
            bits = bin(rem)[2:].count('1')
            
            if bits <= k and k <= rem:
                return k
        
        return -1