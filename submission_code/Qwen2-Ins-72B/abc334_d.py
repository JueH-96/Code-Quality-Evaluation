import bisect

N, Q = map(int, input().split())
R = sorted(list(map(int, input().split())))
S = [0]
for r in R:
    S.append(S[-1] + r)

for _ in range(Q):
    X = int(input())
    i = bisect.bisect_right(S, X)
    print(i-1)