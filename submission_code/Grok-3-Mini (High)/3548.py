import math
import itertools
from typing import List

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        ans = 0
        
        def generate_freq():
            def helper(sum_rem, index, current):
                if index == 10:
                    if sum_rem == 0:
                        yield current[:]
                    return
                for f in range(0, sum_rem + 1):
                    current[index] = f
                    yield from helper(sum_rem - f, index + 1, current)
            return helper(n, 0, [0] * 10)
        
        def is_k_palindromic(freq):
            if n % 2 == 1:
                odd_d = [d for d in range(10) if freq[d] % 2 == 1]
                if len(odd_d) != 1:
                    return False
                M = odd_d[0]
                c = [(freq[d] - 1) // 2 if d == M else freq[d] // 2 for d in range(10)]
            else:
                M = -1
                c = [freq[d] // 2 for d in range(10)]
            m = n // 2
            L = []
            for d in range(10):
                for _ in range(c[d]):
                    L.append(d)
            for p in itertools.permutations(L):
                digits = [0] * n
                for i in range(m):
                    digits[i] = p[i]
                    digits[n - 1 - i] = p[i]
                if n % 2 == 1:
                    digits[m] = M
                if digits[0] > 0:
                    num = 0
                    for dig in digits:
                        num = num * 10 + dig
                    if num % k == 0:
                        return True
            return False
        
        for freq in generate_freq():
            odd_count = sum(1 for f in freq if f % 2 == 1)
            if (n % 2 == 1 and odd_count != 1) or (n % 2 == 0 and odd_count != 0):
                continue
            if is_k_palindromic(freq):
                total_perm = math.factorial(n)
                den = 1
                for f in freq:
                    den *= math.factorial(f)
                total_perm //= den
                if freq[0] > 0:
                    num_zero_first = math.factorial(n - 1)
                    den_zero = math.factorial(freq[0] - 1)
                    for f in freq[1:]:
                        den_zero *= math.factorial(f)
                    num_zero_first //= den_zero
                else:
                    num_zero_first = 0
                num_good = total_perm - num_zero_first
                ans += num_good
        
        return ans