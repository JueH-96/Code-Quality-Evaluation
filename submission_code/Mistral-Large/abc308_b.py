import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Extract the plate colors
C = data[2:2+N]

# Extract the color-price mappings
D = data[2+N:2+N+M]
P = list(map(int, data[2+N+M:2+N+M+M+1]))

# Create a dictionary for the price lookup
price_dict = {D[i]: P[i+1] for i in range(M)}

# Calculate the total price
total_price = 0
for color in C:
    if color in price_dict:
        total_price += price_dict[color]
    else:
        total_price += P[0]

# Output the result
print(total_price)