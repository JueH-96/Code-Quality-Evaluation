import sys

# Set recursion limit for the judge system, as the DP can be deep.
sys.setrecursionlimit(2000)

def main():
    """
    Main function to solve the graph counting problem.
    """
    N, P = map(int, sys.stdin.readline().split())

    k = N // 2
    max_edges_bipartite = k * k
    
    # Precompute factorials and their modular inverses for combinations.
    # The maximum 'n' in nCr(n, r) will be k*k. For N=30, k=15, this is 225.
    max_comb_n = max_edges_bipartite
    fact = [1] * (max_comb_n + 1)
    invfact = [1] * (max_comb_n + 1)
    for i in range(1, max_comb_n + 1):
        fact[i] = (fact[i - 1] * i) % P

    invfact[max_comb_n] = pow(fact[max_comb_n], P - 2, P)
    for i in range(max_comb_n - 1, -1, -1):
        invfact[i] = (invfact[i + 1] * (i + 1)) % P

    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (invfact[r] * invfact[n - r]) % P
        return (num * den) % P

    # --- Polynomial Operations ---
    def poly_add(p1, p2):
        n1, n2 = len(p1), len(p2)
        n = max(n1, n2)
        res = [0] * n
        for i in range(n):
            v1 = p1[i] if i < n1 else 0
            v2 = p2[i] if i < n2 else 0
            res[i] = (v1 + v2) % P
        return res

    def poly_sub(p1, p2):
        n1, n2 = len(p1), len(p2)
        n = max(n1, n2)
        res = [0] * n
        for i in range(n):
            v1 = p1[i] if i < n1 else 0
            v2 = p2[i] if i < n2 else 0
            res[i] = (v1 - v2 + P) % P
        return res

    def poly_mul(p1, p2):
        n1, n2 = len(p1), len(p2)
        if n1 == 0 or n2 == 0:
            return []
        n = n1 + n2 - 1
        res = [0] * n
        for i in range(n1):
            if p1[i] == 0: continue
            for j in range(n2):
                res[i + j] = (res[i + j] + p1[i] * p2[j]) % P
        return res

    def poly_scalar_mul(c, p):
        return [(c * x) % P for x in p]
    
    memo_total_poly = {}
    def get_total_poly(n, r):
        # Returns polynomial for C(n*r, e) for e=0..n*r
        if (n, r) in memo_total_poly:
            return memo_total_poly[(n,r)]
        
        total_edges = n * r
        p = [0] * (total_edges + 1)
        for i in range(total_edges + 1):
            p[i] = nCr_mod(total_edges, i)
        memo_total_poly[(n,r)] = p
        return p

    # dp[i][j] stores the polynomial for connected bipartite graphs on partitions of size i and j.
    dp = [[[] for _ in range(k + 1)] for _ in range(k + 1)]
    
    for i in range(k + 1):
        for j in range(k + 1):
            if i == 0 or j == 0:
                continue

            # Polynomial for all bipartite graphs on i,j partitions.
            total_ij_poly = get_total_poly(i, j)
            
            # Subtract disconnected graphs.
            # Sum over the size (p, q) of the connected component containing a fixed vertex.
            sum_poly = [0]
            for p in range(1, i + 1):
                for q in range(1, j + 1):
                    if p == i and q == j:
                        continue
                    
                    coeff = (nCr_mod(i - 1, p - 1) * nCr_mod(j, q)) % P
                    if coeff == 0:
                        continue
                    
                    # Term is C(i-1,p-1)C(j,q) * dp[p][q] * total[i-p][j-q]
                    term = poly_mul(dp[p][q], get_total_poly(i - p, j - q))
                    term = poly_scalar_mul(coeff, term)
                    sum_poly = poly_add(sum_poly, term)

            dp[i][j] = poly_sub(total_ij_poly, sum_poly)
            
    # Number of ways to choose the partitions: C(N-1, k-1)
    # Vertex 1 is fixed in one partition. Choose k-1 vertices for its partition
    # from the remaining N-1 vertices.
    partition_ways = nCr_mod(N - 1, k - 1)
    
    final_poly = dp[k][k]
    
    # Calculate and format the results for each M.
    results = []
    max_total_edges = N * (N - 1) // 2
    for M in range(N - 1, max_total_edges + 1):
        if M < len(final_poly):
            ans = (partition_ways * final_poly[M]) % P
            results.append(ans)
        else:
            results.append(0)
    
    print(*results)

if __name__ == "__main__":
    main()