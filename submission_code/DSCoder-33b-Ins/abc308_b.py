# YOUR CODE HERE

# Read the inputs
N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

# Create a dictionary to map colors to prices
price_dict = {D[i]: P[i+1] for i in range(M)}

# Initialize the total price
total_price = 0

# Iterate over the plates
for plate in C:
    # If the color of the plate is in the dictionary, add the price to the total
    if plate in price_dict:
        total_price += price_dict[plate]
    # If the color of the plate is not in the dictionary, add the price of the default color to the total
    else:
        total_price += P[0]

# Print the total price
print(total_price)