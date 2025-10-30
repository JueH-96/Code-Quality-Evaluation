# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
Q = int(data[1])

queries = data[2:]

# Initialize the positions of the dragon parts
positions = [(i, 0) for i in range(1, N + 1)]

# Direction vectors
directions = {
    'R': (1, 0),
    'L': (-1, 0),
    'U': (0, 1),
    'D': (0, -1)
}

output = []
index = 0

for _ in range(Q):
    query_type = int(queries[index])
    if query_type == 1:
        direction = queries[index + 1]
        dx, dy = directions[direction]
        
        # Move the head
        new_head_pos = (positions[0][0] + dx, positions[0][1] + dy)
        
        # Move the rest of the body
        for i in range(N - 1, 0, -1):
            positions[i] = positions[i - 1]
        
        positions[0] = new_head_pos
        index += 2
    elif query_type == 2:
        part = int(queries[index + 1]) - 1
        output.append(f"{positions[part][0]} {positions[part][1]}")
        index += 2

# Print all the results for type 2 queries
print("
".join(output))