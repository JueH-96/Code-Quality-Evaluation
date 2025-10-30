class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        from math import factorial
        from collections import Counter

        def generate_palindromes(n):
            palindromes = []
            half = (n + 1) // 2  # Number of digits to generate
            start = 10**(half - 1)
            end = 10**half
            for i in range(start, end):
                s = str(i)
                if n % 2 == 0:
                    pal = s + s[::-1]
                else:
                    pal = s + s[-2::-1]
                palindromes.append(int(pal))
            return palindromes

        palindromes = generate_palindromes(n)
        valid_palindromes = [x for x in palindromes if x % k == 0]

        result = 0
        for x in valid_palindromes:
            s = str(x)
            freq = Counter(s)
            total_perms = factorial(n)
            for count in freq.values():
                total_perms //= factorial(count)

            if '0' not in freq:
                valid_perms = total_perms
            else:
                freq_zero = freq['0'] - 1
                freq_non_zero = freq.copy()
                freq_non_zero['0'] = freq_zero
                perms_zero_first = factorial(n - 1)
                for count in freq_non_zero.values():
                    perms_zero_first //= factorial(count)
                valid_perms = total_perms - perms_zero_first

            result += valid_perms

        return result