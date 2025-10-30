# YOUR CODE HERE
import sys

# Read the input integer N from standard input
n = int(sys.stdin.readline())

# Create a list to store the results of each penalty kick
results = []

# Iterate through each penalty kick from 1 to N
for i in range(1, n + 1):
    # Check if the kick number i is a multiple of 3
    if i % 3 == 0:
        # If it's a multiple of 3, Takahashi fails ('x')
        results.append('x')
    else:
        # Otherwise, Takahashi succeeds ('o')
        results.append('o')

# Join the characters in the list to form the final result string
output_string = "".join(results)

# Print the result string to standard output
print(output_string)