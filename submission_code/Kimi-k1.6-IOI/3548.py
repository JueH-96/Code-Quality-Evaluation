import math
from collections import defaultdict

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if n == 0:
            return 0
        
        m = (n + 1) // 2
        start = 10 ** (m - 1)
        end = 10 ** m - 1
        
        palindromes = []
        for prefix in range(start, end + 1):
            s = str(prefix)
            if n % 2 == 0:
                pal_str = s + s[::-1]
            else:
                pal_str = s + s[:-1][::-1]
            num = int(pal_str)
            palindromes.append(num)
        
        valid_palindromes = [x for x in palindromes if x % k == 0]
        
        digit_counts = set()
        for p in valid_palindromes:
            counts = [0] * 10
            for d in str(p):
                counts[int(d)] += 1
            digit_counts.add(tuple(counts))
        
        result = 0
        for counts in digit_counts:
            c0 = counts[0]
            total_perm = math.factorial(n)
            denominator = 1
            for c in counts:
                denominator *= math.factorial(c)
            total_perm //= denominator
            
            if c0 == 0:
                valid = total_perm
            else:
                valid = total_perm - (total_perm * c0) // n
            
            result += valid
        
        return result