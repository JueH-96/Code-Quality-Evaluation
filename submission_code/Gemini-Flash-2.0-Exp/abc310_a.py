N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

min_dish_price = min(D)
coupon_price = Q + min_dish_price

print(min(P, coupon_price))