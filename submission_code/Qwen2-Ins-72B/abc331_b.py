N, S, M, L = map(int, input().split())

cost = [float('inf')] * (N + 1)
cost[0] = 0

for i in range(N + 1):
    if i + 6 <= N:
        cost[i + 6] = min(cost[i + 6], cost[i] + S)
    if i + 8 <= N:
        cost[i + 8] = min(cost[i + 8], cost[i] + M)
    if i + 12 <= N:
        cost[i + 12] = min(cost[i + 12], cost[i] + L)

print(cost[N])