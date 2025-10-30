# Read the input values
n, m = map(int, input().split())
c = input().split()
d = input().split()
p_values = list(map(int, input().split()))

# Create a dictionary to map each color to its price
price_map = {}
for i in range(m):
    color = d[i]
    price = p_values[i + 1]
    price_map[color] = price

# Calculate the total price
total = 0
p0 = p_values[0]
for color in c:
    if color in price_map:
        total += price_map[color]
    else:
        total += p0

# Print the result
print(total)