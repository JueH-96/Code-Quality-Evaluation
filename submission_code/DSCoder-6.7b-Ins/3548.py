from math import comb

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if k == 1: return 10**n
        if n == 1: return 9
        res = 0
        for i in range((n+1)//2, n+1):
            res += 2*sum(comb(9*(i-j), j) for j in range((n-i)//2+1))
        return res