# YOUR CODE HERE
N, M = map(int, input().split())
P = list(map(int, input().split()))
L = list(map(int, input().split()))
D = list(map(int, input().split()))

# Create a list of (discount, item_index, coupon_index) for all valid pairs
pairs = []
for i in range(N):
    for j in range(M):
        if P[i] >= L[j]:
            pairs.append((D[j], i, j))

# Sort by discount in descending order
pairs.sort(reverse=True)

# Greedy assignment
used_items = [False] * N
used_coupons = [False] * M
total_discount = 0

for discount, item_idx, coupon_idx in pairs:
    if not used_items[item_idx] and not used_coupons[coupon_idx]:
        used_items[item_idx] = True
        used_coupons[coupon_idx] = True
        total_discount += discount

# Calculate total cost
total_cost = sum(P) - total_discount
print(total_cost)