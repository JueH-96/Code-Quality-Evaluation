# YOUR CODE HERE
N, M = map(int, input().split())
P = list(map(int, input().split()))
L = list(map(int, input().split()))
D = list(map(int, input().split()))

# Create list of (price, index) and sort by price descending
items = [(P[i], i) for i in range(N)]
items.sort(reverse=True)

# Create list of (discount, min_price, index) and sort by discount descending
coupons = [(D[i], L[i], i) for i in range(M)]
coupons.sort(reverse=True)

# Track which coupons have been used
used_coupons = [False] * M

# Calculate total cost
total_cost = sum(P)  # Start with full price for all items

# For each item (from highest price to lowest)
for price, item_idx in items:
    best_discount = 0
    best_coupon_idx = -1
    
    # Find the best unused coupon that can be applied to this item
    for j in range(M):
        if not used_coupons[j] and coupons[j][1] <= price:
            if coupons[j][0] > best_discount:
                best_discount = coupons[j][0]
                best_coupon_idx = j
    
    # If we found a valid coupon, use it
    if best_coupon_idx != -1:
        total_cost -= best_discount
        used_coupons[best_coupon_idx] = True

print(total_cost)