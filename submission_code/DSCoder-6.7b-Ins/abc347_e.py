def solve():
    N, Q = map(int, input().split())
    queries = list(map(int, input().split()))
    A = [0] * N
    S = set()

    for query in queries:
        if query in S:
            S.remove(query)
        else:
            S.add(query)
        if query-1 < N:
            A[query-1] += len(S)

    print(' '.join(map(str, A)))

solve()