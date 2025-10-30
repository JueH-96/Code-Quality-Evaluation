from itertools import permutations

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        max_fact = 20
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i - 1] * i
        
        def generate_g(rem, dim):
            if dim == 1:
                yield [rem]
            else:
                for i in range(rem + 1):
                    for res in generate_g(rem - i, dim - 1):
                        yield [i] + res
        
        def process(f):
            if sum(f[1:]) == 0:
                return 0
            center_digit = None
            if n % 2 == 1:
                for d in range(10):
                    if f[d] % 2 == 1:
                        center_digit = d
                        break
            digits = []
            for d in range(10):
                digits.extend([d] * (f[d] // 2))
            L = len(digits)
            good = False
            if L == 0:
                if center_digit % k == 0:
                    good = True
                else:
                    good = False
            else:
                if all(d == 0 for d in digits):
                    good = False
                else:
                    perms_set = set(permutations(digits))
                    good = False
                    for p in perms_set:
                        if p[0] == 0:
                            continue
                        if n % 2 == 0:
                            full_digits = list(p) + list(p)[::-1]
                        else:
                            full_digits = list(p) + [center_digit] + list(p)[::-1]
                        num = 0
                        for digit in full_digits:
                            num = num * 10 + digit
                        if num % k == 0:
                            good = True
                            break
            if not good:
                return 0
            total_perms = fact[n]
            for cnt in f:
                total_perms //= fact[cnt]
            subtract = 0
            if f[0] > 0:
                subtract = fact[n - 1]
                for d in range(10):
                    cnt_val = f[d]
                    if d == 0:
                        subtract //= fact[cnt_val - 1]
                    else:
                        subtract //= fact[cnt_val]
            return total_perms - subtract
        
        ans = 0
        if n % 2 == 0:
            doubletons = n // 2
            for g in generate_g(doubletons, 10):
                f_vec = [2 * x for x in g]
                ans += process(f_vec)
        else:
            doubletons = (n - 1) // 2
            for center in range(10):
                for g in generate_g(doubletons, 10):
                    f_vec = [2 * x for x in g]
                    f_vec[center] += 1
                    ans += process(f_vec)
        return ans