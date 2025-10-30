# YOUR CODE HERE
import sys

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = int(data[1])
K = int(data[2])
PQ = list(map(int, data[3:]))

# Calculate total price of products
total_price = 0
for i in range(N):
    P = PQ[2 * i]
    Q = PQ[2 * i + 1]
    total_price += P * Q

# Determine shipping fee
if total_price >= S:
    shipping_fee = 0
else:
    shipping_fee = K

# Calculate total amount to pay
total_amount = total_price + shipping_fee

# Print the result
print(total_amount)