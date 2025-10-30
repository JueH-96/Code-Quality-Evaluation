N, M = map(int, input().split())
C = [input() for _ in range(N)]
D = [input() for _ in range(M)]
P = list(map(int, input().split()))

total_price = 0
for c in C:
    if c in D:
        total_price += P[D.index(c) + 1]
    else:
        total_price += P[0]

print(total_price)