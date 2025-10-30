import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A_values = list(map(int, data[index:index + N]))

# Create adjacency list (1-based indexing)
adj = [0] + A_values

# Create visited array (1-based indexing)
visited = [False] * (N + 1)

# Iterate through each node
for start in range(1, N + 1):
    if not visited[start]:
        path = []
        curr = start
        # Traverse the path until a visited node is found
        while not visited[curr]:
            visited[curr] = True
            path.append(curr)
            curr = adj[curr]
        # Check if the current node is in the path (cycle detected)
        if curr in path:
            # Find the index where the cycle starts
            idx = path.index(curr)
            cycle = path[idx:]
            M = len(cycle)
            # Output the cycle
            print(M)
            print(*cycle)
            sys.exit()  # Exit after finding and printing the cycle