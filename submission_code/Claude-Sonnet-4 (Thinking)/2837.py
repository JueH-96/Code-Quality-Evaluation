class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 51):  # Check up to 50 operations
            target = num1 - k * num2
            if target <= 0:
                continue
            popcount = bin(target).count('1')
            if popcount <= k <= target:
                return k
        return -1