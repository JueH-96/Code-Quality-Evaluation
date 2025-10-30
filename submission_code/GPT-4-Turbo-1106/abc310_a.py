# Read the input values
N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

# Calculate the minimum total amount with and without the coupon
min_with_coupon = Q + min(D)
min_without_coupon = P

# Print the minimum of the two calculated amounts
print(min(min_with_coupon, min_without_coupon))