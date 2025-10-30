import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

queries = data[2:]

# Initialize the positions of the dragon parts
positions = [[i, 0] for i in range(1, N + 1)]

# Deque to store the movements
movements = deque()

# Process each query
index = 0
output = []

for query in queries:
    if query == '1':
        direction = data[index + 2]
        index += 2

        # Move the head
        if direction == 'R':
            positions[0][0] += 1
        elif direction == 'L':
            positions[0][0] -= 1
        elif direction == 'U':
            positions[0][1] += 1
        elif direction == 'D':
            positions[0][1] -= 1

        # Store the movement
        movements.append((positions[0][0], positions[0][1]))

        # Update the positions of the other parts
        for i in range(1, N):
            positions[i] = movements.popleft()
            movements.append(positions[i])

    elif query == '2':
        part = int(data[index + 2])
        index += 2

        # Get the position of the specified part
        output.append(f"{positions[part - 1][0]} {positions[part - 1][1]}")

# Print the output
sys.stdout.write("
".join(output) + "
")