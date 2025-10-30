# Read the first line and split into R, G, B
line1 = input()
parts = line1.split()
R = int(parts[0])
G = int(parts[1])
B = int(parts[2])

# Read the disliked color
C = input().strip()

# Create a dictionary mapping colors to their prices
colors = {
    "Red": R,
    "Green": G,
    "Blue": B
}

# Exclude the disliked color and find the minimum price
prices = [colors[col] for col in colors if col != C]
min_price = min(prices)

# Print the minimum price
print(min_price)