class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        for k in range(61):
            S = num1 - k * num2
            if S < 0:
                break  # S is decreasing or constant, no need to check further k
            popcount_S = bin(S).count('1')
            max_power_val = 1 << 60
            ceil_div = (S + max_power_val - 1) // max_power_val
            if k >= max(popcount_S, ceil_div) and S >= k:
                return k
        return -1