# Read input
N, S, K = map(int, input().split())
products = []
for _ in range(N):
    p, q = map(int, input().split())
    products.append((p, q))

# Calculate total price
total_price = 0
for p, q in products:
    total_price += p * q

# Calculate shipping fee
if total_price >= S:
    shipping_fee = 0
else:
    shipping_fee = K

# Calculate total amount
total_amount = total_price + shipping_fee

# Print the result
print(total_amount)