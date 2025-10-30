class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        def count_bits(x):
            return bin(x).count('1')
        
        k = 1
        while True:
            target = num1 - k * num2
            if target < 0:
                return -1
            if count_bits(target) <= k <= target:
                return k
            k += 1