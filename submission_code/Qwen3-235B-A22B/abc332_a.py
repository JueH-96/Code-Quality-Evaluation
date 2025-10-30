# Read the first line and split into N, S, K
n, s, k = map(int, input().split())

total_products = 0

# Process each product line
for _ in range(n):
    p, q = map(int, input().split())
    total_products += p * q

# Determine shipping cost
shipping = 0 if total_products >= s else k

# Calculate and print the total amount
print(total_products + shipping)