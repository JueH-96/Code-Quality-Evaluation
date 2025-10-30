# YOUR CODE HERE
import sys

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
P = int(data[1])
Q = int(data[2])
D = list(map(int, data[3:3+N]))

# Calculate the minimum cost
min_cost = min(P, Q + min(D))

# Print the result
print(min_cost)