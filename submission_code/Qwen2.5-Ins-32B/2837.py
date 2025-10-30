class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for operations in range(1, 61):
            target = num1 - operations * num2
            if target >= 0 and target.bit_count() <= operations <= target:
                return operations
        return -1