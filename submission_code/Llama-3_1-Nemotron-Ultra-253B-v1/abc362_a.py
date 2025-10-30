# Read the input values
r, g, b = map(int, input().split())
c = input().strip()

# List of prices corresponding to each color
prices = [r, g, b]
color_list = ['Red', 'Green', 'Blue']

# Determine the index of the excluded color
excluded_index = color_list.index(c)

# Collect the prices of the allowed colors
allowed_prices = [prices[i] for i in range(3) if i != excluded_index]

# Find and print the minimum price
print(min(allowed_prices))