# Read the first line and split into N, P, Q
N, P, Q = map(int, input().split())

# Read the second line and split into list of D_i
D = list(map(int, input().split()))

# Find the minimum D_i
min_D = min(D)

# Calculate the total cost if using the coupon
total_with_coupon = Q + min_D

# Find the minimum between paying P directly or using the coupon
minimal_total = min(P, total_with_coupon)

# Print the result
print(minimal_total)