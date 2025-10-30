# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse the input
N = int(data[0])
M = int(data[1])
P = int(data[2])

# Calculate the number of full moon days between day 1 and day N, inclusive
full_moon_count = (N - M) // P + 1 if M <= N else 0

# Print the result
print(full_moon_count)