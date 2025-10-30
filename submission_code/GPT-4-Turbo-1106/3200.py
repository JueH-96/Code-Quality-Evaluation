class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        
        # Helper function to calculate factorial modulo MOD
        def factorial_mod(x):
            result = 1
            for i in range(1, x + 1):
                result = (result * i) % MOD
            return result
        
        # Helper function to calculate modular inverse
        def mod_inverse(x):
            return pow(x, MOD - 2, MOD)
        
        # Helper function to calculate nCr modulo MOD
        def nCr_mod(n, r):
            if r > n:
                return 0
            return (factorial_mod(n) * mod_inverse(factorial_mod(r)) % MOD * mod_inverse(factorial_mod(n - r)) % MOD) % MOD
        
        # If n is less than 4, it's not possible to form "leet"
        if n < 4:
            return 0
        
        # Calculate the number of permutations of "leet" in a string of length n
        # which is n! / (n-4)! since we are fixing 4 positions for "leet"
        permutations_of_leet = factorial_mod(n) * mod_inverse(factorial_mod(n - 4)) % MOD
        
        # Calculate the number of ways to fill the remaining positions with any lowercase letters
        # which is 26^(n-4) since each of the remaining n-4 positions can be any of the 26 letters
        remaining_positions = pow(26, n - 4, MOD)
        
        # The total number of good strings is the product of the two counts above
        total_good_strings = (permutations_of_leet * remaining_positions) % MOD
        
        return total_good_strings

# Example usage:
# sol = Solution()
# print(sol.stringCount(4))  # Output: 12
# print(sol.stringCount(10)) # Output: 83943898