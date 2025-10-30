# Read N, S, K from the first line of input
N, S, K = map(int, input().split())

# Initialize total_product_price to 0
total_product_price = 0

# Loop N times to read the price and quantity for each product type
for _ in range(N):
    # Read P_i and Q_i for the current product type
    P, Q = map(int, input().split())
    
    # Add the cost of this product type (price * quantity) to the total
    total_product_price += P * Q

# Determine the shipping fee based on the total product price
shipping_fee = 0
if total_product_price < S:
    # If total product price is less than S, apply the shipping fee K
    shipping_fee = K

# Calculate the final amount to be paid
total_amount_paid = total_product_price + shipping_fee

# Print the result
print(total_amount_paid)