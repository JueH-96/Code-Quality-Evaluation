class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):  # k is the number of operations
            target = num1 - k * num2
            if target <= 0:
                return -1
            if target.bit_count() <= k <= target:
                return k
        return -1