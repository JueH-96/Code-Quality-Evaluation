import heapq

N, M = map(int, input().split())
P = list(map(int, input().split()))
L = list(map(int, input().split()))
D = list(map(int, input().split()))

coupons = sorted([(L[i], D[i]) for i in range(M)], reverse=True)
prices = sorted(P, reverse=True)

total = sum(prices)
j = 0
for i in range(N):
    while j < M and prices[i] < coupons[j][0]:
        j += 1
    if j < M and prices[i] >= coupons[j][0]:
        total -= min(prices[i], coupons[j][1])
        j += 1
    if j >= M:
        break
print(total)