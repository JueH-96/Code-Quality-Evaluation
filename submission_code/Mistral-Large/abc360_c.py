import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
W = list(map(int, data[N+1:2*N+1]))

# Create a list to track the positions of items
position = [0] * (N + 1)
for i in range(1, N + 1):
    position[A[i-1]] = i

# Initialize the total cost
total_cost = 0

# Create a visited list to track visited boxes
visited = [False] * (N + 1)

# Iterate over each box
for i in range(1, N + 1):
    if visited[i] or position[i] == i:
        continue

    # Initialize the cycle cost and start a new cycle
    cycle_cost = 0
    cycle_start = i
    current = i

    # Traverse the cycle
    while True:
        visited[current] = True
        next_box = position[current]
        cycle_cost += W[next_box - 1]
        position[current] = current
        current = next_box

        if current == cycle_start:
            break

    # Add the minimum cycle cost to the total cost
    total_cost += min(cycle_cost + W[cycle_start - 1], cycle_cost + W[position[cycle_start] - 1])

# Print the total cost
print(total_cost)