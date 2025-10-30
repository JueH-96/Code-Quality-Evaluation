N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

regular_price = P
min_coupon_price = float('inf')

for dish_price in D:
    coupon_price = Q + dish_price
    min_coupon_price = min(min_coupon_price, coupon_price)

print(min(regular_price, min_coupon_price))