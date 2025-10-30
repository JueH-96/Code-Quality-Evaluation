# YOUR CODE HERE
import sys
import sys
import sys
def solve():
    import sys, math
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353

    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    
    from collections import defaultdict

    tree = defaultdict(list)
    roots = []
    for i in range(1, N+1):
        if A[i-1] ==0:
            roots.append(i)
        else:
            tree[A[i-1]].append(i)
    
    # Precompute factorial and inverse factorial
    max_n = N
    factorial = [1]*(max_n +1)
    for i in range(1, max_n +1):
        factorial[i] = factorial[i-1]*i % MOD
    inv_fact = [1]*(max_n +1)
    inv_fact[max_n] = pow(factorial[max_n], MOD-2, MOD)
    for i in range(max_n,0,-1):
        inv_fact[i-1] = inv_fact[i]*i % MOD

    # Compute subtree sizes
    sizes = []
    def dfs(u):
        size =1
        for v in tree[u]:
            size += dfs(v)
        sizes.append(math.factorial(size) % MOD)
        return size
    for r in roots:
        dfs(r)
    
    # Number of roots
    k = len(roots)
    
    # Compute n! / (prod(s_i! ) *k! )
    numerator = factorial[N]
    denominator = 1
    for s in sizes:
        denominator = denominator * s % MOD
    denominator = denominator * factorial[k] % MOD
    answer = numerator * pow(denominator, MOD-2, MOD) % MOD
    print(answer)