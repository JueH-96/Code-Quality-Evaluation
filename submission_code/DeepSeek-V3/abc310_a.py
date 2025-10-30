# Read input values
N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

# Calculate the minimum cost when using the coupon
min_dish = min(D)
total_with_coupon = Q + min_dish

# Compare with the regular price
min_total = min(total_with_coupon, P)

# Print the result
print(min_total)