import sys
input = sys.stdin.read
MOD = 998244353

def solve():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    A = [0] * (N + 1)
    for i in range(2, N + 1):
        A[i] = int(data[index])
        index += 1
    
    queries = []
    for _ in range(Q):
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        queries.append((u, v))
    
    # Precompute factorials and factorial inverses
    factorial = [1] * (N + 1)
    for i in range(2, N + 1):
        factorial[i] = factorial[i - 1] * i % MOD
    
    # Precompute factorial inverses using Fermat's little theorem
    inv_factorial = [1] * (N + 1)
    inv_factorial[N] = pow(factorial[N], MOD - 2, MOD)
    for i in range(N - 1, 0, -1):
        inv_factorial[i] = inv_factorial[i + 1] * (i + 1) % MOD
    
    # Precompute powers of (N-1)!
    fact_pow = [1] * (N + 1)
    fact_pow[1] = factorial[N - 1]
    for i in range(2, N + 1):
        fact_pow[i] = fact_pow[i - 1] * factorial[N - 1] % MOD
    
    # Answer queries
    results = []
    for u, v in queries:
        if u > v:
            u, v = v, u
        
        # Sum of distances from u to v over all trees
        # Distance is always A[v] since u < v and the only path in any tree is direct from u to v
        # There are (N-1)! trees
        distance_sum = A[v] * fact_pow[1] % MOD
        results.append(distance_sum)
    
    # Output all results
    sys.stdout.write('
'.join(map(str, results)) + '
')