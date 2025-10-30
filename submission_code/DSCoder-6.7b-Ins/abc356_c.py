from itertools import product, combinations

def solve():
    N, M, K = map(int, input().split())
    tests = [list(map(int, input().split())) for _ in range(M)]

    real_keys = set(range(1, N+1))
    dp = [set() for _ in range(N+1)]
    dp[0] = set([frozenset()])

    for i in range(1, N+1):
        for combo in dp[i-1]:
            dp[i].add(combo)
            dp[i].add(combo.union(frozenset([i])))

    valid_combos = []
    for combo in dp[N]:
        if len(combo) >= K:
            valid_combos.append(combo)

    valid_results = 0
    for test in tests:
        C, *A, R = test
        keys = set(A)
        if len(keys & real_keys) < K:
            continue
        for i in range(K, len(keys)+1):
            for combo in combinations(keys, i):
                if combo in valid_combos:
                    if R == 'o':
                        valid_results += 1
                    break

    print(valid_results)

solve()