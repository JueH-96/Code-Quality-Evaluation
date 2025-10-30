class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        if n < 4:
            return 0
        
        # Calculate factorials and factorial inverses up to n using modular arithmetic
        fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        def modinv(x):
            # Fermat's little theorem: a^(p-1) ≡ 1 (mod p) => a^(p-2) ≡ a^(-1) (mod p)
            return pow(x, MOD - 2, MOD)
        
        inv_fact = [1] * (n + 1)
        for i in range(2, n + 1):
            inv_fact[i] = modinv(fact[i])
        
        # Function to calculate nCr % MOD
        def nCr(n, r):
            if r > n or r < 0:
                return 0
            return fact[n] * inv_fact[r] % MOD * inv_fact[n - r] % MOD
        
        # Total number of strings of length n
        total_strings = pow(26, n, MOD)
        
        # Calculate the number of good strings containing "leet"
        # We need to place "leet" in any order in the string of length n
        # There are 4! ways to arrange "leet"
        ways_to_arrange_leet = fact[4]
        
        # Count the number of ways to place "leet" in a string of length n
        # We place "leet" and then fill the rest (n-4) positions with any of the 26 letters
        good_strings = ways_to_arrange_leet * pow(26, n - 4, MOD) % MOD
        
        return good_strings