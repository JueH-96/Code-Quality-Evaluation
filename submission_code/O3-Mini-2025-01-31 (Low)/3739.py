MOD = 10**9 + 7

# Precompute factorials and inverse factorials up to a given max_n.
def precompute_factorials(max_n):
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
    for i in range(max_n, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % MOD
    return fact, inv_fact

def nCr(n, r, fact, inv_fact):
    if r < 0 or r > n:
        return 0
    return fact[n] * inv_fact[r] % MOD * inv_fact[n-r] % MOD

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        # total number of cells
        total_cells = m * n

        # We need to compute C(total_cells - 2, k - 2) modulo MOD.
        # Precompute factorials up to total_cells.
        fact, inv_fact = precompute_factorials(total_cells)
        comb = nCr(total_cells - 2, k - 2, fact, inv_fact)
        
        # Sum of pairwise Manhattan distances: we can separate row and column components.
        # For rows: For each pair of distinct rows, the vertical contribution is (row2 - row1).
        # And for any such row pair, there are n choices for the column of the first cell and n for the second -> n*n pairs.
        # The sum of differences for rows indexed 0...m-1 is:
        #     sum_{i<j} (j - i). A known formula: (m*(m-1)*(m+1))//6.
        # Similarly for columns: (n*(n-1)*(n+1))//6 multiplied by m*m.

        # Compute row contribution:
        row_contrib = n * n % MOD
        row_contrib = row_contrib * (m * (m - 1) % MOD * (m + 1) % MOD) % MOD
        row_contrib = row_contrib * pow(6, MOD - 2, MOD) % MOD   # division by 6 mod MOD

        # Compute column contribution:
        col_contrib = m * m % MOD
        col_contrib = col_contrib * (n * (n - 1) % MOD * (n + 1) % MOD) % MOD
        col_contrib = col_contrib * pow(6, MOD - 2, MOD) % MOD

        total_pair_dist = (row_contrib + col_contrib) % MOD

        # For each pair of cells, they participate in exactly C(total_cells - 2, k - 2) arrangements where those two cells are chosen (since we need to choose k - 2 out of the remaining cells).
        answer = total_pair_dist * comb % MOD
        return answer