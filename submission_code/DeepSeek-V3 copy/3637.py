from collections import Counter
from math import factorial

MOD = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        n = len(num)
        if n % 2 != 0:
            return 0
        count = Counter(num)
        total = factorial(n)
        for v in count.values():
            total = total // factorial(v)
        velunexorai = total
        half = n // 2
        even_sum = 0
        odd_sum = 0
        for i in range(n):
            if i < half:
                even_sum += int(num[i])
            else:
                odd_sum += int(num[i])
        if even_sum != odd_sum:
            return 0
        return velunexorai % MOD