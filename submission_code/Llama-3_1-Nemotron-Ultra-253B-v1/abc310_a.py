# Read input
N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

# Calculate the minimum cost when using the coupon
min_dish = min(D)
coupon_cost = Q + min_dish

# The answer is the minimum between not using the coupon and using it
print(min(P, coupon_cost))