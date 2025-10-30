# YOUR CODE HERE
# Read the number of product types (N), the shipping fee threshold (S),
# and the shipping fee (K) from the first line of input.
N, S, K = map(int, input().split())

# Initialize a variable to store the total price of the products.
total_price = 0

# Loop N times to read the details for each product type.
# The underscore (_) is used as a variable for the loop counter
# when its value is not needed inside the loop.
for _ in range(N):
    # Read the price (P) and quantity (Q) for the current product type.
    P, Q = map(int, input().split())
    
    # Calculate the cost for this product type and add it to the running total.
    total_price += P * Q

# Check if the total price of the products is less than the threshold S.
if total_price < S:
    # If the total price is less than S, add the shipping fee K.
    total_price += K

# Print the final total amount to be paid.
print(total_price)