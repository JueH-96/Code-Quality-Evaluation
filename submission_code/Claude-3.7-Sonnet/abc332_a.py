# Read the first line containing N, S, K
N, S, K = map(int, input().split())

# Initialize the total price of products
total_product_price = 0

# Process each product
for _ in range(N):
    P, Q = map(int, input().split())
    total_product_price += P * Q

# Calculate the shipping fee
shipping_fee = K if total_product_price < S else 0

# Calculate the total amount to pay
total_amount = total_product_price + shipping_fee

# Print the result
print(total_amount)