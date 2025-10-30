# YOUR CODE HERE
N, M = map(int, input().split())
P = list(map(int, input().split()))
L = list(map(int, input().split()))
D = list(map(int, input().split()))

coupons = []
for i in range(M):
    coupons.append((L[i], D[i]))

coupons.sort(key=lambda x: x[1], reverse=True)
P.sort()

total_cost = sum(P)
used_coupons = [False] * M
for i in range(M):
    best_item = -1
    best_discount = -1
    for j in range(N):
        if P[j] >= coupons[i][0] and coupons[i][1] > best_discount:
            best_discount = coupons[i][1]
            best_item = j
    if best_item != -1:
        total_cost -= best_discount

print(total_cost)