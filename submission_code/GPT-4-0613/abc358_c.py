from itertools import combinations

N, M = map(int, input().split())
S = [input() for _ in range(N)]

min_stands = N
for r in range(1, N+1):
    for comb in combinations(range(N), r):
        flavors = set()
        for i in comb:
            flavors.update(j for j in range(M) if S[i][j] == 'o')
        if len(flavors) == M:
            min_stands = min(min_stands, r)
            break
    if min_stands < N:
        break
print(min_stands)