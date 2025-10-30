class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num2 == - (1 << 60):
            return -1
        max_k = 120
        for K in range(1, max_k +1):
            S = num1 - K * num2
            if S < K:
                continue
            pop = bin(S).count('1')
            if pop <= K:
                return K
        return -1