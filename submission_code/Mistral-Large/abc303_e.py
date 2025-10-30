import sys
from collections import deque

def find_stars(N, edges):
    # Initialize the adjacency list
    adj = [[] for _ in range(N+1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Initialize the queue for BFS and the visited list
    queue = deque([1])
    visited = [False] * (N+1)
    visited[1] = True

    # List to store the sizes of the stars
    stars = []

    while queue:
        current = queue.popleft()
        neighbors = adj[current]

        # If the current node has exactly one neighbor, it's a leaf node
        if len(neighbors) == 1:
            star_size = 1
            leaf_node = current
            center_node = neighbors[0]

            # Traverse to find the size of the star
            while center_node != 1 and len(adj[center_node]) == 2:
                next_node = adj[center_node][0] if adj[center_node][1] == leaf_node else adj[center_node][1]
                leaf_node = center_node
                center_node = next_node
                star_size += 1

            stars.append(star_size)

            # Mark all nodes in this star as visited
            center_node = neighbors[0]
            while center_node != 1 and len(adj[center_node]) == 2:
                visited[center_node] = True
                next_node = adj[center_node][0] if adj[center_node][1] == leaf_node else adj[center_node][1]
                leaf_node = center_node
                center_node = next_node

        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    # Sort and print the sizes of the stars
    stars.sort()
    print(" ".join(map(str, stars)))

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
edges = []
for i in range(1, N):
    u, v = int(data[2*i-1]), int(data[2*i])
    edges.append((u, v))

# Find and print the stars
find_stars(N, edges)