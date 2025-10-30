# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

# Convert the string S to a list of integers
A = [int(char) for char in S]

# Initialize the result
result = 0

# Initialize the current value
current = A[0]

# Iterate through the sequence
for i in range(1, N):
    # Update the current value using the NAND operation
    current = 1 if current == A[i] else 0
    # Add the current value to the result
    result += current

# Add the sum of the sequence itself
result += sum(A)

# Print the result
print(result)