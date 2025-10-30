class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):  # Check k from 1 to 60 inclusive
            S = num1 - num2 * k
            if S <= 0:
                continue
            if S >= k and bin(S).count('1') <= k:
                return k
        return -1