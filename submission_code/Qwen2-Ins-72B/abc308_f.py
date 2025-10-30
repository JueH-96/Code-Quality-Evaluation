import heapq

N, M = map(int, input().split())
P = list(map(int, input().split()))
L = list(map(int, input().split()))
D = list(map(int, input().split()))

P.sort()
D = [d for l, d in sorted(zip(L, D), reverse=True)]
D += [0] * N

ans = sum(P)
i = 0
for p in P:
    if p - D[i] < 0:
        break
    ans -= D[i]
    i += 1

print(ans)