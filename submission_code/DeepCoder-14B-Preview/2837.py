class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        def count_set_bits(x):
            return bin(x).count('1')
        
        if num2 == 0:
            return count_set_bits(num1)
        elif num2 > 0:
            if (num1 - num2) >= 1:
                S = num1 - num2
                if (S & (S - 1)) == 0 and S != 0:
                    return 1
            return -1
        else:
            for k in range(1, 10**6 + 1):
                S = num1 - k * num2
                if S < k or S < 0:
                    continue
                if count_set_bits(S) <= k:
                    return k
            return -1