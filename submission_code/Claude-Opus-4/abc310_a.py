# YOUR CODE HERE
N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

# Option 1: Pay regular price
regular_price = P

# Option 2: Use coupon (pay Q for drink + cheapest dish)
coupon_price = Q + min(D)

# Print the minimum
print(min(regular_price, coupon_price))