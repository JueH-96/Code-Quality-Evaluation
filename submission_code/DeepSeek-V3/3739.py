import math

MOD = 10**9 + 7

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        if k < 2:
            return 0
        
        total_cells = m * n
        if k > total_cells:
            return 0
        
        # Precompute factorial, inverse factorial modulo MOD up to total_cells
        max_n = total_cells
        fact = [1] * (max_n + 1)
        inv_fact = [1] * (max_n + 1)
        
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        def comb(a, b):
            if a < 0 or b < 0 or a < b:
                return 0
            return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD
        
        c_total = comb(total_cells, k)
        if k == 2:
            c_pair = 1
        else:
            c_pair = comb(total_cells - 2, k - 2)
        
        # Compute x contribution: sum_{i < p} (p - i) * n^2 * c_pair
        # Sum over all i and p (i < p) of (p - i) multiplied by the number of column pairs (each column pair is same for x)
        # So total x contribution is sum_{i < p} (p -i) * m * n * (m *n -1 choose k-2) * (number of column pairs? No. Wait, for each pair of cells (i,j) and (p,q), the x contribution is |i-p|. So for all possible j and q (columns), it's |i-p| * 1. So for each i and p, the term is |i-p| * n * n (since j and q can be any columns). But wait, for each i and p, there are n possible j's and n possible q's, but in the arrangement, only two specific cells are chosen. So the number of pairs of cells with rows i and p is n^2 * (for each j and q, but in the arrangement, only two are selected. So the correct count is for each pair of distinct cells (i,j) and (p,q), the x contribution is |i-p|. The total x contribution is sum_{i,j} sum_{p,q} |i-p| * [number of arrangements where these two are selected]. Which is |i-p| * C(total_cells - 2, k-2). Similarly for y.
        
        # So total x contribution is sum_{i=1 to m} sum_{p=1 to m} sum_{j=1 to n} sum_{q=1 to n} |i-p| * C(M*N -2, k-2) for i,j and p,q being distinct.
        # But since i and p can be the same, but the Manhattan distance would be zero, so only i ≠ p contributes. But wait, the Manhattan distance between (i,j) and (p,q) is |i-p| + |j-q|. So for the x component, it's |i-p|, regardless of j and q. So the total x contribution is sum_{i,p} |i-p| * n^2 * C(M*N -2, k-2) when i ≠ p. But wait, the pairs are unordered, and each pair of cells is counted once. So for each pair of distinct cells (i,j) and (p,q), the x contribution is |i-p|. The number of such pairs is for all possible i,j,p,q with (i,j) ≠ (p,q). But the total number of terms is total_cells * (total_cells - 1). But for each i and p, the number of j and q pairs is n^2. So the total x contribution is sum_{i=1 to m} sum_{p=1 to m} |i-p| * n^2 * C(M*N-2, k-2). But this is for all i and p, including i = p. When i = p, the x contribution is zero. So the sum can be written as sum_{i,p} |i-p| * n^2 * C(M*N-2, k-2). But this counts each pair twice (once for (i,p) and once for (p,i)), but since |i-p| is the same, it's okay. So total x contribution is sum_{i < p} 2*(p -i) * n^2 * C(M*N-2, k-2).
        
        # So compute Sx = sum_{i < p} (p -i) for all i < p in 1..m. Then multiply by 2 * n^2 * C(M*N-2, k-2).
        # Similarly for Sy.
        
        # Compute Sx
        prefix = [0] * (m + 1)
        for i in range(1, m + 1):
            prefix[i] = (prefix[i-1] + i) % MOD
        
        Sx = 0
        for i in range(1, m + 1):
            # Sum_{p=i+1 to m} (p -i) = sum_{d=1 to m-i} d = (m-i)(m-i+1)/2
            cnt = m - i
            sum_p = cnt * (cnt + 1) // 2
            Sx += sum_p
        Sx %= MOD
        Sx = Sx * n * n % MOD
        
        # Compute Sy
        prefix = [0] * (n + 1)
        for j in range(1, n + 1):
            prefix[j] = (prefix[j-1] + j) % MOD
        
        Sy = 0
        for j in range(1, n + 1):
            cnt = n - j
            sum_q = cnt * (cnt + 1) // 2
            Sy += sum_q
        Sy %= MOD
        Sy = Sy * m * m % MOD
        
        total = (Sx + Sy) % MOD
        c_pair = comb(total_cells - 2, k - 2)
        total = total * c_pair % MOD
        return total