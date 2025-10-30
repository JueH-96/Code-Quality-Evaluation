import sys
from collections import deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
Q = int(data[index + 1])
index += 2

# Initialize the positions of the dragon parts
positions = [(i, 0) for i in range(1, N + 1)]
# Initialize the deque to keep track of the positions
dragon = deque(positions)

# Function to move the head and update the positions of the parts
def move_head(direction):
    head_x, head_y = dragon[0]
    if direction == 'R':
        new_head = (head_x + 1, head_y)
    elif direction == 'L':
        new_head = (head_x - 1, head_y)
    elif direction == 'U':
        new_head = (head_x, head_y + 1)
    elif direction == 'D':
        new_head = (head_x, head_y - 1)
    dragon.appendleft(new_head)
    dragon.pop()

# Process the queries
results = []
for _ in range(Q):
    query_type = int(data[index])
    if query_type == 1:
        direction = data[index + 1]
        move_head(direction)
        index += 2
    elif query_type == 2:
        p = int(data[index + 1])
        results.append(dragon[p - 1])
        index += 2

# Print the results
for result in results:
    print(result[0], result[1])