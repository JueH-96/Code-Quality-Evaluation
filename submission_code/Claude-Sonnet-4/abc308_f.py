# YOUR CODE HERE
N, M = map(int, input().split())
P = list(map(int, input().split()))
L = list(map(int, input().split()))
D = list(map(int, input().split()))

# Start with all items at regular price
total_cost = sum(P)

# Create list of (discount, min_price) for coupons
coupons = [(D[i], L[i]) for i in range(M)]
# Sort by discount in descending order
coupons.sort(reverse=True)

# Sort items by price in descending order, keep track of original indices
items_with_idx = [(P[i], i) for i in range(N)]
items_with_idx.sort(reverse=True)

used_items = [False] * N

# For each coupon (best discount first)
for discount, min_price in coupons:
    # Find the most expensive item that can use this coupon and hasn't been used
    for price, item_idx in items_with_idx:
        if not used_items[item_idx] and price >= min_price:
            # Use this coupon on this item
            used_items[item_idx] = True
            total_cost -= discount
            break

print(total_cost)