# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = list(map(int, data[1:]))

# Initialize the result list with zeros
result = [0] * N

# Use a stack to keep track of buildings
stack = []

for i in range(N):
    # While the stack is not empty and the current building is taller than the building at the index stored in the stack
    while stack and H[i] > H[stack[-1]]:
        # Pop the index from the stack
        idx = stack.pop()
        # The number of buildings visible from the popped index is the current index minus the popped index minus one
        result[idx] = i - idx - 1
    # Push the current index onto the stack
    stack.append(i)

# For the remaining buildings in the stack, the number of buildings visible is 0
while stack:
    idx = stack.pop()
    result[idx] = 0

# Print the result
print(" ".join(map(str, result)))