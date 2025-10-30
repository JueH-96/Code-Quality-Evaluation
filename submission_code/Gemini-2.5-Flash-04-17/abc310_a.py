# Read N, P, Q
n, p, q = map(int, input().split())

# Read the list of dish prices D
d_prices = list(map(int, input().split()))

# Find the minimum dish price
min_dish_price = min(d_prices)

# Calculate the cost if using the coupon (Q for drink + minimum dish price)
cost_with_coupon = q + min_dish_price

# Calculate the cost if not using the coupon (regular price P)
cost_without_coupon = p

# The minimum total cost is the minimum of the two options
minimum_total_cost = min(cost_with_coupon, cost_without_coupon)

# Print the result
print(minimum_total_cost)