# Read the first line and split into N, P, Q
N, P, Q = map(int, input().split())

# Read the second line and split into list of dish prices D
D = list(map(int, input().split()))

# Find the minimum dish price
min_dish = min(D)

# Calculate the minimum cost with coupon
min_with_coupon = Q + min_dish

# Determine the overall minimum cost
total_min = min(P, min_with_coupon)

# Print the result
print(total_min)