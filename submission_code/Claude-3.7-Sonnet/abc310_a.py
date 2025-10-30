# Read input
N, P, Q = map(int, input().split())
dishes = list(map(int, input().split()))

# Option 1: Buy the drink at regular price
regular_price = P

# Option 2: Use the coupon (discounted drink + cheapest dish)
min_dish_price = min(dishes)
coupon_price = Q + min_dish_price

# Choose the cheaper option
min_payment = min(regular_price, coupon_price)

# Output result
print(min_payment)