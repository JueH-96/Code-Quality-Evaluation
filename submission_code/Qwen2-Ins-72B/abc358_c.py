from itertools import combinations

N, M = map(int, input().split())
S = [input() for _ in range(N)]

ans = float('inf')
for i in range(1, N+1):
    for comb in combinations(S, i):
        tmp = ''.join(['x']*M)
        for s in comb:
            for j in range(M):
                if s[j] == 'o' and tmp[j] == 'x':
                    tmp = tmp[:j] + 'o' + tmp[j+1:]
        if 'x' not in tmp:
            ans = min(ans, i)

print(ans)