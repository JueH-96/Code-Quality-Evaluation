# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Sort the monsters by their strength
A.sort()

# Initialize the total experience points
total_points = 0

# Iterate through the monsters
for i in range(N):
    # If the monster is even-numbered, add its strength twice
    if (i + 1) % 2 == 0:
        total_points += 2 * A[i]
    # Otherwise, add its strength once
    else:
        total_points += A[i]

# Print the total experience points
print(total_points)