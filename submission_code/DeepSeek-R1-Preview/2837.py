class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(1, 61):
            S = num1 - k * num2
            if S <= 0:
                continue
            if S < k:
                continue
            set_bits = bin(S).count('1')
            if set_bits <= k:
                return k
        a = num2 + 1
        if a < 0:
            return 61
        else:
            if num1 >= 61 * a:
                return 61
            else:
                return -1