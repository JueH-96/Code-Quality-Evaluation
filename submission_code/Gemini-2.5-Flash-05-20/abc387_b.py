# Read the input integer X
X = int(input())

# Initialize a variable to store the sum of integers that are not X
sum_not_X = 0

# Iterate through the rows (i) from 1 to 9
for i in range(1, 10):
    # Iterate through the columns (j) from 1 to 9
    for j in range(1, 10):
        # Calculate the value in the current cell
        current_value = i * j
        
        # Check if the current value is not equal to X
        if current_value != X:
            # If it's not X, add it to our sum
            sum_not_X += current_value

# Print the final sum
print(sum_not_X)