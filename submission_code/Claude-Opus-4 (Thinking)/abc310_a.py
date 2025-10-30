# YOUR CODE HERE
N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

# Option 1: Regular price
regular_price = P

# Option 2: Coupon price + cheapest dish
coupon_price = Q + min(D)

# Print the minimum
print(min(regular_price, coupon_price))