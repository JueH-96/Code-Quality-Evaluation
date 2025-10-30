N, M = map(int, input().split())
P = list(map(int, input().split()))
L = list(map(int, input().split()))
D = list(map(int, input().split()))

# Sort coupons by discount in descending order
coupons = sorted(range(M), key=lambda i: D[i], reverse=True)

used_items = [False] * N
total_discount = 0

for coupon_idx in coupons:
    best_item = -1
    best_price = float('inf')
    
    # Find the cheapest applicable item that hasn't been used
    for item_idx in range(N):
        if not used_items[item_idx] and P[item_idx] >= L[coupon_idx] and P[item_idx] < best_price:
            best_item = item_idx
            best_price = P[item_idx]
    
    if best_item != -1:
        used_items[best_item] = True
        total_discount += D[coupon_idx]

total_cost = sum(P) - total_discount
print(total_cost)