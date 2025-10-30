# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse the input
N = int(data[0])
L = int(data[1])
A = list(map(int, data[2:]))

# Count the number of people who passed the exam
passed_count = sum(1 for score in A if score >= L)

# Print the result
print(passed_count)