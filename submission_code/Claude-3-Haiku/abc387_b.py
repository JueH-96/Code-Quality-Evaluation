# Read the input from stdin
x = int(input())

# Initialize the sum to 0
total_sum = 0

# Iterate through the 9x9 grid
for i in range(1, 10):
    for j in range(1, 10):
        # Calculate the value in the current cell
        value = i * j
        
        # If the value is not equal to x, add it to the total sum
        if value != x:
            total_sum += value

# Print the total sum to stdout
print(total_sum)