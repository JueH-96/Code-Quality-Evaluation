import sys
from itertools import zip_longest

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

P = list(map(int, data[2:2+N]))
L = list(map(int, data[2+N:2+N+M]))
D = list(map(int, data[2+N+M:2+N+M+M]))

# Sort items by price in ascending order
P.sort()

# Sort coupons by discount value in descending order
coupons = sorted(zip(L, D), key=lambda x: x[1], reverse=True)

total_cost = 0
coupon_index = 0

for price in P:
    # Find the first coupon that can be applied to the current item
    while coupon_index < M and (coupons[coupon_index][0] > price or coupons[coupon_index][1] == 0):
        coupon_index += 1

    if coupon_index < M:
        # Apply the coupon and mark it as used
        total_cost += price - coupons[coupon_index][1]
        coupons[coupon_index] = (coupons[coupon_index][0], 0)
    else:
        # No applicable coupon, buy at full price
        total_cost += price

print(total_cost)