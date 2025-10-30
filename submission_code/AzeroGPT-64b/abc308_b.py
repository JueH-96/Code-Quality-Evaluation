from collections import defaultdict

N, M = map(int, input().split())
C = [input() for _ in range(N)]
D = [input() for _ in range(M)]
P = list(map(int, input().split()))

prices = defaultdict(lambda: P[0])
for i in range(M):
    prices[D[i]] = P[i + 1]

total_price = sum(prices[c] for c in C)
print(total_price)