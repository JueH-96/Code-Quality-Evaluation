class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 201):
            S = num1 - k * num2
            if num2 > 0 and S < k:
                break
            if S < k:
                continue
            if bin(S).count('1') <= k:
                return k
        return -1