# YOUR CODE HERE
# Read N, S, K
N, S, K = map(int, input().split())

# Calculate total cost of products
total_cost = 0
for i in range(N):
    P, Q = map(int, input().split())
    total_cost += P * Q

# Determine shipping fee
if total_cost >= S:
    shipping_fee = 0
else:
    shipping_fee = K

# Calculate final amount to pay
final_amount = total_cost + shipping_fee

print(final_amount)