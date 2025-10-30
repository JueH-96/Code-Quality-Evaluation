# YOUR CODE HERE
n, p, q = map(int, input().split())
dishes = list(map(int, input().split()))

# Option 1: Pay regular price
regular_cost = p

# Option 2: Use coupon + cheapest dish
coupon_cost = q + min(dishes)

# Print minimum cost
print(min(regular_cost, coupon_cost))