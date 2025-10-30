class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7

        # Helper function to calculate (a^b) % MOD
        def pow_mod(a, b):
            result = 1
            base = a
            while b > 0:
                if b % 2 == 1:
                    result = (result * base) % MOD
                base = (base * base) % MOD
                b //= 2
            return result

        # Total number of strings of length n
        total_strings = pow_mod(26, n)

        # Subtract the number of strings that do not contain any of the characters 'l', 'e', 't'
        # These are strings made up of the remaining 23 characters
        bad_strings = pow_mod(23, n)

        # The number of good strings is the total number of strings minus the bad strings
        good_strings = (total_strings - bad_strings + MOD) % MOD

        return good_strings