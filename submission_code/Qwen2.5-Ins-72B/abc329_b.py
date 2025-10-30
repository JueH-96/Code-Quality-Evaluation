# YOUR CODE HERE
import sys

# Read input
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
A = list(map(int, data[1:]))

# Find the largest and second largest numbers
largest = max(A)
second_largest = max(x for x in A if x != largest)

# Print the second largest number
print(second_largest)