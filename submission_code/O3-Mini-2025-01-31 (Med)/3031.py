from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345

        # Extended Euclidean Algorithm to compute modular inverse
        def egcd(a, b):
            if b == 0:
                return a, 1, 0
            g, x1, y1 = egcd(b, a % b)
            return g, y1, x1 - (a // b) * y1

        def modinv(a, m):
            # a and m are coprime (this will be true for our "remainder" parts)
            g, x, _ = egcd(a, m)
            if g != 1:
                # Should not happen if a is coprime with m.
                return None
            else:
                return x % m

        # We'll factorize each cell with respect to primes 3, 5, and 823
        # since 12345 = 3 * 5 * 823
        def factorize(x):
            cnt3 = cnt5 = cnt823 = 0
            while x % 3 == 0:
                cnt3 += 1
                x //= 3
            while x % 5 == 0:
                cnt5 += 1
                x //= 5
            while x % 823 == 0:
                cnt823 += 1
                x //= 823
            return x, cnt3, cnt5, cnt823  # remaining part is coprime to MOD

        n = len(grid)
        m_len = len(grid[0]) if n > 0 else 0

        # We will accumulate two kinds of information:
        # 1) The "remainder" product tot_rem which is the product of each cell's remainder (after removing factors 3, 5, 823) modulo MOD.
        # 2) The global exponents for the factors 3, 5, and 823.
        tot_rem = 1
        tot_3 = tot_5 = tot_823 = 0

        # Also store each cell's factorization result in a matrix "fact"
        fact = [[None] * m_len for _ in range(n)]
        for i in range(n):
            for j in range(m_len):
                rem, c3, c5, c823 = factorize(grid[i][j])
                fact[i][j] = (rem, c3, c5, c823)
                tot_rem = (tot_rem * (rem % MOD)) % MOD
                tot_3 += c3
                tot_5 += c5
                tot_823 += c823

        # Now, for each cell, the product of all elements except grid[i][j] equals:
        #     (global_product / grid[i][j]) mod MOD.
        # We calculate:
        # global_product = tot_rem * (3^(tot_3)) * (5^(tot_5)) * (823^(tot_823))
        # grid[i][j] = (rem * 3^(c3) * 5^(c5) * 823^(c823))
        # Thus, after cancellation:
        # p[i][j] = (tot_rem * inv(rem) mod MOD) *
        #           (3^(tot_3 - c3) mod MOD) *
        #           (5^(tot_5 - c5) mod MOD) *
        #           (823^(tot_823 - c823) mod MOD)  mod MOD.
        res = [[0] * m_len for _ in range(n)]
        for i in range(n):
            for j in range(m_len):
                rem, c3, c5, c823 = fact[i][j]
                # Since rem is coprime with MOD (we removed any factors 3,5,823),
                # we can safely compute its modular inverse.
                inv_rem = modinv(rem, MOD)
                part_rem = (tot_rem * inv_rem) % MOD
                part3 = pow(3, tot_3 - c3, MOD)
                part5 = pow(5, tot_5 - c5, MOD)
                part823 = pow(823, tot_823 - c823, MOD)
                res[i][j] = (part_rem * part3 % MOD * part5 % MOD * part823) % MOD

        return res