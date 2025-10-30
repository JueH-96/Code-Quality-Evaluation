from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        mod = 12345
        # The mod value factors into primes: 12345 = 3 * 5 * 823.
        # For each grid element, we want to factor out all factors of 3, 5, and 823.
        # Then we can represent grid[i][j] as: r * 3^(a) * 5^(b) * 823^(c),
        # where r is co-prime with mod. The product over all elements becomes:
        # global_rem * (3^(sum a)) * (5^(sum b)) * (823^(sum c)) mod mod.
        # And for p[i][j] we want:
        # p[i][j] = (global product excluding grid[i][j])
        #         = (global_rem * inv(r)) * (3^(global_a - a)) * (5^(global_b - b)) * (823^(global_c - c)) mod mod.
        
        primes = [3, 5, 823]
        n = len(grid)
        m = len(grid[0])
        # global factors from all cells
        global_factors = {p: 0 for p in primes}
        # global remainder (product of the "coprime" parts) modulo mod
        global_rem = 1
        
        # We will precompute for each cell its reduced remainder and its counts for factors 3, 5, and 823.
        # Since grid has up to 1e5 elements, storing extra info is acceptable.
        cell_info = [[None] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                x = grid[i][j]
                fac_counts = {p: 0 for p in primes}
                # Factor out each of the three mod factors.
                for p in primes:
                    while x % p == 0:
                        fac_counts[p] += 1
                        x //= p
                # After removing factors of 3, 5, and 823, x is co-prime with mod.
                # Compute the remainder part modulo mod.
                r = x % mod
                cell_info[i][j] = (r, fac_counts)
                global_rem = (global_rem * r) % mod
                for p in primes:
                    global_factors[p] += fac_counts[p]
        
        # Now compute the product matrix p.
        # For each cell, p[i][j] = (overall product excluding grid[i][j]) mod mod.
        # That equals to:
        #    (global_rem * modular_inverse(cell_r)) mod mod
        #    multiplied by each prime factor p raised to the power global_factors[p] - cell_factors[p], all mod mod.
        res = [[0] * m for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                r, fac_counts = cell_info[i][j]
                # Since r is co-prime with mod, the modular inverse exists.
                inv_r = pow(r, -1, mod)
                val = (global_rem * inv_r) % mod
                for p in primes:
                    exp = global_factors[p] - fac_counts[p]
                    # Multiply by p^exp mod mod.
                    val = (val * pow(p, exp, mod)) % mod
                res[i][j] = val
        
        return res