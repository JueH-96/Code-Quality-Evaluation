# Read the input from stdin
import sys

# Read the number of products, the minimum total price for free shipping, and the shipping fee
N, S, K = map(int, sys.stdin.readline().split())

# Initialize the total price
total_price = 0

# Read the price and quantity of each product
for _ in range(N):
    P, Q = map(int, sys.stdin.readline().split())
    # Add the price of the current product to the total price
    total_price += P * Q

# Calculate the shipping fee
shipping_fee = 0 if total_price >= S else K

# Calculate the total amount to pay
total_amount = total_price + shipping_fee

# Print the total amount to pay
print(total_amount)