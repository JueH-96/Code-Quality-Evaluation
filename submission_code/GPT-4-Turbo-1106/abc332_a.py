# Read the first line of input
N, S, K = map(int, input().split())

# Initialize the total price
total_price = 0

# Read the price and quantity of each product and add to the total price
for _ in range(N):
    P_i, Q_i = map(int, input().split())
    total_price += P_i * Q_i

# Calculate the shipping fee
shipping_fee = 0 if total_price >= S else K

# Calculate the total amount to pay
total_amount = total_price + shipping_fee

# Print the total amount to pay
print(total_amount)