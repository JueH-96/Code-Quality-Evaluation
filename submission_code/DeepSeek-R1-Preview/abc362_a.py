# Read the pen prices
R, G, B = map(int, input().split())
# Read the disliked color
C = input().strip()

# Create a dictionary mapping colors to their prices
color_prices = {
    'Red': R,
    'Green': G,
    'Blue': B
}

# Collect the prices of colors that are not disliked
available_prices = [price for color, price in color_prices.items() if color != C]

# Find and print the minimum price
print(min(available_prices))