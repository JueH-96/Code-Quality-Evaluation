# Read the first line of input
n, s, k = map(int, input().split())

# Initialize total product price
total_products = 0

# Read each product's price and quantity, accumulate the total
for _ in range(n):
    p, q = map(int, input().split())
    total_products += p * q

# Determine shipping cost
shipping = 0 if total_products >= s else k

# Calculate and print the final amount
print(total_products + shipping)