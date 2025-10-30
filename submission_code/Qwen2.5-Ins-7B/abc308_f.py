# YOUR CODE HERE
n, m = map(int, input().split())
prices = list(map(int, input().split()))
coupons = [tuple(map(int, input().split())) for _ in range(m)]

coupons.sort(key=lambda x: x[1])
prices.sort()

total_cost = 0
used_coupons = set()

for i in range(n):
    if i < m and prices[i] >= coupons[i][0] and coupons[i][1] not in used_coupons:
        total_cost += prices[i] - coupons[i][1]
        used_coupons.add(coupons[i][1])
    else:
        total_cost += prices[i]

print(total_cost)