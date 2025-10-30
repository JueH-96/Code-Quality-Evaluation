# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Create a list to store the final order of people
order = [-1] * N

# Initialize a stack to keep track of people waiting to be placed
stack = []

# Iterate through the given arrangement
for i in range(N):
    if A[i] == -1:
        # If the person is at the front, place them at the current position
        order[i] = i + 1
        # Place all people waiting behind them in the correct order
        while stack:
            order[stack.pop()] = i + 1
    else:
        # If the person is behind someone, add them to the stack
        stack.append(i)

# Print the final order of people
print(' '.join(map(str, order)))