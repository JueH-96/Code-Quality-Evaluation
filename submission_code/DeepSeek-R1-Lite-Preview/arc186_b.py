MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))

    # Build the tree
    children = [[] for _ in range(N)]
    roots = []
    for i in range(N):
        if A[i] == 0:
            roots.append(i)
        else:
            children[A[i]-1].append(i)

    # Precompute factorial and inverse factorial
    fact = [1] * (N+1)
    for i in range(1, N+1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (N+1)
    inv_fact[N] = pow(fact[N], MOD-2, MOD)
    for i in range(N-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    # DP to compute number of ways for each subtree
    dp = [1] * N
    size = [1] * N

    def dfs(u):
        for v in children[u]:
            dfs(v)
            size[u] += size[v]
            dp[u] = dp[u] * dp[v] % MOD
            dp[u] = dp[u] * fact[size[v]] % MOD

    for r in roots:
        dfs(r)
        dp[r] = dp[r] * fact[size[r]-1] % MOD

    # Compute total ways
    total = 1
    for r in roots:
        total = total * dp[r] % MOD

    # Compute multinomial coefficient
    multinom = fact[N]
    for r in roots:
        multinom = multinom * inv_fact[size[r]] % MOD
    total = total * multinom % MOD

    # Divide by number of ways to arrange roots
    num_roots = len(roots)
    total = total * inv_fact[num_roots] % MOD

    print(total)

if __name__ == '__main__':
    main()