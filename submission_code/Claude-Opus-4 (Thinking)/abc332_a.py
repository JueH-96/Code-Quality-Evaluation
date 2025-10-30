# YOUR CODE HERE
# Read N, S, K
N, S, K = map(int, input().split())

# Calculate total price of products
total_price = 0
for _ in range(N):
    P, Q = map(int, input().split())
    total_price += P * Q

# Determine shipping fee
if total_price >= S:
    shipping_fee = 0
else:
    shipping_fee = K

# Calculate and print total amount
total_amount = total_price + shipping_fee
print(total_amount)