import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Initialize index to keep track of position in data list
index = 0

# Read N and M
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Read the list of colors Takahashi ate (C_list)
C_list = data[index:index + N]
index += N

# Read the list of special colors (D_list)
D_list = data[index:index + M]
index += M

# Read the price strings and convert to integers
P_str = data[index:index + M + 1]
P = [int(p) for p in P_str]

# Create a dictionary for special color prices
price_dict = {}
for i in range(M):
    price_dict[D_list[i]] = P[i + 1]

# Calculate the total price
total = 0
for color in C_list:
    if color in price_dict:
        total += price_dict[color]
    else:
        total += P[0]

# Output the total
print(total)