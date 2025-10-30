import sys

# Read input
N, M = map(int, input().split())
P = list(map(int, input().split()))
L = list(map(int, input().split()))
D = list(map(int, input().split()))

# Sort coupons in descending order of discount
coupons = sorted(zip(L, D), reverse=True)

# Initialize total cost
total_cost = 0

# Iterate through items
for p in P:
    # Check if there is a coupon that can be used for this item
    for i, (l, d) in enumerate(coupons):
        if p >= l:
            total_cost += p - d
            coupons.pop(i)
            break
    else:
        total_cost += p

# Print the answer
print(total_cost)