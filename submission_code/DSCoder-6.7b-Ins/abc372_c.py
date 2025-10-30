import sys
from collections import defaultdict

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    queries = [tuple(sys.stdin.readline().split()) for _ in range(Q)]

    prefix_count = defaultdict(int)
    suffix_count = defaultdict(int)
    total_count = defaultdict(int)

    for i in range(N):
        total_count[S[i]] += 1
        if i != 0:
            prefix_count[S[i]] = prefix_count[S[i-1]]
        if i != N-1:
            suffix_count[S[i+1]] = suffix_count[S[i]]
        if i > 0 and i < N-1:
            prefix_count[S[i]] += 1
            suffix_count[S[i+1]] += 1
            total_count[S[i]] -= 1

    for i in range(Q):
        X, C = queries[i]
        X = int(X)
        if X > 1:
            prefix_count[S[X-1]] -= 1
            total_count[S[X-1]] -= 1
            if X < N-1:
                suffix_count[S[X+1]] -= 1
        if X < N:
            suffix_count[S[X]] -= 1
            total_count[S[X]] += 1
            if X > 0:
                prefix_count[S[X-1]] += 1
        S = S[:X] + C + S[X+1:]
        print(prefix_count['A'] + suffix_count['B'] + total_count['C'] - (prefix_count['B'] + suffix_count['C'] + total_count['A']))

solve()