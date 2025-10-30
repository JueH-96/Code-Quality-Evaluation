class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7

        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res

        def nCr(n, r):
            if r < 0 or r > n:
                return 0
            if r == 0 or r == n:
                return 1
            if r > n // 2:
                r = n - r

            numerator = 1
            for i in range(r):
                numerator = (numerator * (n - i)) % MOD

            denominator = 1
            for i in range(1, r + 1):
                denominator = (denominator * i) % MOD

            return (numerator * power(denominator, MOD - 2)) % MOD

        total_strings = power(26, n)

        count_l_0 = power(25, n)
        count_e_lt_2 = (power(25, n) + (n * power(25, n - 1)) % MOD) % MOD
        count_t_0 = power(25, n)

        count_l_0_e_lt_2 = (power(24, n) + (n * power(24, n - 1)) % MOD) % MOD
        count_l_0_t_0 = power(24, n)
        count_e_lt_2_t_0 = (power(24, n) + (n * power(24, n - 1)) % MOD) % MOD

        count_l_0_e_lt_2_t_0 = (power(23, n) + (n * power(23, n - 1)) % MOD) % MOD

        complement = (count_l_0 + count_e_lt_2 + count_t_0) % MOD
        complement = (complement - count_l_0_e_lt_2 - count_l_0_t_0 - count_e_lt_2_t_0) % MOD
        complement = (complement + count_l_0_e_lt_2_t_0) % MOD

        complement = (complement + MOD) % MOD

        good_strings = (total_strings - complement + MOD) % MOD
        return good_strings