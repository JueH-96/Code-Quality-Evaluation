class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(0, 62):
            S = num1 - k * num2
            if S < k:
                if num2 >= 0:
                    break
                else:
                    continue
            if bin(S).count('1') <= k:
                return k
        return -1