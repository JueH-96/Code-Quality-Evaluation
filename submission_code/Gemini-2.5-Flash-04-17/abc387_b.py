import sys

# Read the integer X from standard input
X = int(sys.stdin.readline())

# Initialize a variable to store the sum of numbers not equal to X
sum_not_X = 0

# Iterate through the 9x9 multiplication table
# i represents the row number (from 1 to 9)
for i in range(1, 10):
    # j represents the column number (from 1 to 9)
    for j in range(1, 10):
        # Calculate the product for the current cell
        product = i * j
        
        # If the product is not equal to X, add it to the sum
        if product != X:
            sum_not_X += product

# Print the final sum
print(sum_not_X)