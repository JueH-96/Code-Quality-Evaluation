import sys

# Read the line from standard input
line = sys.stdin.readline().split()

# Convert the parts to integers
A = int(line[0])
B = int(line[1])

# Calculate the sum
sum_ab = A + B

# Calculate the square of the sum
result = sum_ab * sum_ab

# Print the result to standard output
print(result)