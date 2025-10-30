# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize variables to keep track of the total experience points and the count of defeated monsters
total_experience = 0
defeated_count = 0

# Iterate through each monster
for i in range(N):
    # If the monster's strength is greater than 0, defeat it
    if A[i] > 0:
        total_experience += A[i]
        defeated_count += 1
        # If it is an even-numbered defeated monster, add additional experience points
        if defeated_count % 2 == 0:
            total_experience += A[i]

# Print the maximum total experience points
print(total_experience)