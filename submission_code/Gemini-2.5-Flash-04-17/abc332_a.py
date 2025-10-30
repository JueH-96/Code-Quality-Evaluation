# Read N, S, K from the first line
n, s, k = map(int, input().split())

# Initialize total product price
total_product_price = 0

# Read product details and calculate total price
for _ in range(n):
    p, q = map(int, input().split())
    total_product_price += p * q

# Determine shipping fee
if total_product_price < s:
    shipping_fee = k
else:
    shipping_fee = 0

# Calculate total amount paid
total_amount_paid = total_product_price + shipping_fee

# Print the result
print(total_amount_paid)