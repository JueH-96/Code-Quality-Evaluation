# Read input
N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

# Calculate the minimum possible cost using the coupon
min_dish = min(D)
coupon_cost = Q + min_dish

# Determine the minimum total cost
total_cost = min(P, coupon_cost)

# Output the result
print(total_cost)