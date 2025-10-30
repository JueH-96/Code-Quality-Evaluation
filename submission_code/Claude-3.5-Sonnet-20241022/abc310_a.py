N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

# Regular price without coupon
min_cost = P

# Try using coupon with each dish
for dish_price in D:
    # Total cost = discounted drink price + dish price
    coupon_cost = Q + dish_price
    min_cost = min(min_cost, coupon_cost)

print(min_cost)