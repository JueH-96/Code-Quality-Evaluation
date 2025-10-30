class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):
            if num2 > 0:
                T = num1 - k * num2
                if T < k or T < 0:
                    continue
            else:
                T = num1 + k * (-num2)
            if bin(T).count('1') <= k:
                return k
        return -1