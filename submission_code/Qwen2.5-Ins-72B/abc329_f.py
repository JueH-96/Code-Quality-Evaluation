# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
C = list(map(int, data[2:N+2]))
queries = [list(map(int, data[i].split())) for i in range(N+2, N+2+Q)]

# Initialize the boxes with sets to keep track of unique colors
boxes = [set() for _ in range(N+1)]
for i in range(1, N+1):
    boxes[i].add(C[i-1])

# Process the queries
results = []
for a, b in queries:
    # Move all balls from box a to box b
    boxes[b].update(boxes[a])
    boxes[a].clear()
    # Print the number of different colors in box b
    results.append(len(boxes[b]))

# Print the results
for result in results:
    print(result)