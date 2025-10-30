class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7

        # Function to calculate factorial modulo MOD
        def factorial(x):
            result = 1
            for i in range(2, x + 1):
                result = (result * i) % MOD
            return result

        # Function to calculate modular inverse using Fermat's Little Theorem
        def mod_inverse(x):
            return pow(x, MOD - 2, MOD)

        # Function to calculate combinations C(n, k) modulo MOD
        def comb(n, k):
            if k > n:
                return 0
            numerator = factorial(n)
            denominator = (factorial(k) * factorial(n - k)) % MOD
            return (numerator * mod_inverse(denominator)) % MOD

        # Total number of ways to arrange 'l', 'e', 'e', 't' in n positions
        total_ways = 0

        # We need at least 4 positions for 'l', 'e', 'e', 't'
        if n < 4:
            return 0

        # Calculate the number of ways to place 'l', 'e', 'e', 't' in n positions
        for i in range(4, n + 1):
            total_ways = (total_ways + comb(n, i) * comb(i, 2) * comb(i - 2, 1) * comb(i - 3, 1)) % MOD

        return total_ways