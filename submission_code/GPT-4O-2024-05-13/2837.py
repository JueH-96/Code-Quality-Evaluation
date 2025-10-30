class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        import math
        
        def count_bits(x):
            return bin(x).count('1')
        
        for k in range(61):
            target = num1 - k * num2
            if target >= 0 and count_bits(target) <= k and k <= target:
                return k
        
        return -1