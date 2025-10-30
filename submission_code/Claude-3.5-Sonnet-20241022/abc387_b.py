# Read input
X = int(input())

# Initialize sum
total = 0

# Go through each cell in the 9x9 grid
for i in range(1, 10):
    for j in range(1, 10):
        # Calculate the product at current cell
        product = i * j
        # If the product is not X, add it to total
        if product != X:
            total += product

# Print result
print(total)