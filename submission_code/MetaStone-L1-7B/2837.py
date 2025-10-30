class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num2 == 0:
            count = bin(num1).count('1')
            if num1 >= count:
                return count
            else:
                return -1
        elif num2 > 0:
            k_max = num1 // (num2 + 1)
            if k_max < 1:
                return -1
            for k in range(1, k_max + 1):
                S = num1 - k * num2
                if S < k:
                    continue
                count = bin(S).count('1')
                if count <= k:
                    return k
            return -1
        else:
            m = -num2
            if m == 1:
                for k in range(1, 10**6 + 1):
                    S = num1 + k
                    count = bin(S).count('1')
                    if count <= k:
                        return k
                return -1
            else:
                for k in range(1, 10**6 + 1):
                    S = num1 + m * k
                    count = bin(S).count('1')
                    if count <= k:
                        return k
                return -1