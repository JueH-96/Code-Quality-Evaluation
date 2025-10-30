import sys

def power(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

def modInverse(n, m):
    return power(n, m - 2, m)

def precompute_factorials(n, m):
    fact = [1] * (n + 1)
    invFact = [1] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % m
    invFact[n] = modInverse(fact[n], m)
    for i in range(n - 1, 1, -1):
        invFact[i] = (invFact[i + 1] * (i + 1)) % m
    invFact[0] = 1 # Define invFact[0] as 1!^-1 = 1
    invFact[1] = 1 # Define invFact[1] as 1!^-1 = 1
    return fact, invFact

def nCr_mod_p(n, r, fact, invFact, m):
    if r < 0 or r > n:
        return 0
    return (((fact[n] * invFact[r]) % m) * invFact[n - r]) % m

MOD = 998244353

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Necessary conditions for strong connectivity
    # Vertex 1 must be Black to receive a backward edge.
    # Vertex 2N must be White to send a forward edge.
    # From analysis, S_1 must be 'B' and S_2N must be 'W'.
    if S[0] == 'W' or S[-1] == 'B':
        print(0)
        return

    # Count the number of adjacent vertices with the same color
    # This is the number of indices i (1 <= i < 2N) such that S_i == S_{i+1}
    # Using 0-based indexing for string S: indices 0 to 2N-2.
    consecutive_same_color = 0
    for i in range(2 * N - 1):
        if S[i] == S[i+1]:
            consecutive_same_color += 1

    # The number of ways to partition the vertices such that the final graph
    # is strongly connected is given by the formula (N-1)! * C(N-1, K) mod MOD,
    # where K is the number of adjacent same-colored vertices in the string S.
    # This formula comes from a known result for related problems involving paths and matchings.

    # Precompute factorials and inverse factorials up to N
    # Need up to N-1 for factorial and combinations, but precomputing up to N is fine.
    fact, invFact = precompute_factorials(N, MOD)

    # Calculate combinations(N-1, consecutive_same_color)
    combinations = nCr_mod_p(N - 1, consecutive_same_color, fact, invFact, MOD)

    # Calculate (N-1)!
    factorial_N_minus_1 = fact[N - 1]

    # Result is (N-1)! * combinations(N-1, consecutive_same_color) mod MOD
    ans = (factorial_N_minus_1 * combinations) % MOD

    print(ans)

solve()