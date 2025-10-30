# YOUR CODE HERE
# Read N, S, K from the first line
N, S, K = map(int, input().split())

# Initialize total_product_price
total_product_price = 0

# Loop N times to read P_i, Q_i for each product type
# and accumulate the total product price
for _ in range(N):
    P, Q = map(int, input().split())
    total_product_price += P * Q

# Determine the shipping fee and calculate the final amount.
# The shipping fee is 0 yen if the total price of the products
# purchased is S yen or above, and K yen otherwise.
# So, if total_product_price is less than S, add K. Otherwise, add 0.

if total_product_price < S:
    final_amount = total_product_price + K
else:
    # Shipping fee is 0 if total_product_price >= S
    final_amount = total_product_price

# Print the final amount
print(final_amount)