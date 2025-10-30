class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        mod = 12345
        primes = [3, 5, 823]
        n = len(grid)
        m = len(grid[0])
        # We will store for each cell the counts for each prime and the remainder (coprime part)
        cell_prime_counts = [[None]*m for _ in range(n)]
        cell_rem = [[None]*m for _ in range(n)]
        # global factors: total exponents for each prime over the entire grid,
        # and the product of all remainders (numbers that are coprime with mod)
        totalExp = {p:0 for p in primes}
        fullOthers = 1  # product of the remainder parts (mod mod)

        # Process each element in grid:
        for i in range(n):
            for j in range(m):
                num = grid[i][j]
                counts = {}
                for p in primes:
                    count = 0
                    while num % p == 0:
                        num //= p
                        count += 1
                    counts[p] = count
                    totalExp[p] += count
                cell_prime_counts[i][j] = counts
                cell_rem[i][j] = num  # num now is coprime with mod since we removed factors
                fullOthers = (fullOthers * num) % mod

        # Now build the answer matrix.
        # For each cell, the product excluding cell grid[i][j] equals:
        #   ( (product of all remainder parts) / (remainder part of cell) ) *
        #       (3^(totalExp[3]-cellExp[3]) * 5^(totalExp[5]-cellExp[5]) * 823^(totalExp[823]-cellExp[823]) )
        # all computed modulo mod.
        # Note that since cell_rem[i][j] is coprime with mod, modular inverse exists.
        ans = [[0]*m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                # compute the remainder part after excluding cell (using modular inverse)
                rem_val = cell_rem[i][j]
                inv_rem = pow(rem_val, -1, mod)
                part = (fullOthers * inv_rem) % mod

                # incorporate the prime factors that were removed
                for p in primes:
                    exp_excluded = totalExp[p] - cell_prime_counts[i][j][p]
                    # Multiply in p^exp_excluded mod mod
                    part = (part * pow(p, exp_excluded, mod)) % mod
                ans[i][j] = part

        return ans