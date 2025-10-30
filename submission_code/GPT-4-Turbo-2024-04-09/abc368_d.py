import sys
from collections import deque, defaultdict
input = sys.stdin.read

def solve():
    data = input().split()
    idx = 0
    N = int(data[idx])
    K = int(data[idx + 1])
    idx += 2
    
    # Adjacency list for the tree
    tree = defaultdict(list)
    
    # Reading edges
    for _ in range(N - 1):
        A = int(data[idx])
        B = int(data[idx + 1])
        idx += 2
        tree[A].append(B)
        tree[B].append(A)
    
    # Reading the important vertices
    important_vertices = list(map(int, data[idx:idx + K]))
    
    # To find the minimum subtree containing all important vertices, we need to find the Steiner Tree
    # for these vertices in the tree, which is a known NP-hard problem in general graphs but can be
    # solved efficiently in trees using dynamic programming on trees or using the concept of Lowest
    # Common Ancestors (LCA) and distances.
    
    # We will use a BFS from all important vertices simultaneously to find the minimum subtree.
    
    # Distance from any important vertex (multi-source BFS)
    dist = [-1] * (N + 1)
    queue = deque(important_vertices)
    for v in important_vertices:
        dist[v] = 0
    
    # Perform BFS from all important vertices
    while queue:
        node = queue.popleft()
        for neighbor in tree[node]:
            if dist[neighbor] == -1:  # Not visited
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)
    
    # To find the minimum subtree, we need to find the "deepest" node in the BFS tree that is
    # still part of the minimum subtree containing all important vertices.
    
    # We will use a second BFS from the deepest node found in the first BFS to determine the
    # actual minimum subtree.
    max_dist = max(dist)
    farthest_node = dist.index(max_dist)
    
    # Second BFS to determine the subtree size
    queue = deque([farthest_node])
    visited = [False] * (N + 1)
    visited[farthest_node] = True
    subtree_size = 0
    
    while queue:
        node = queue.popleft()
        subtree_size += 1
        for neighbor in tree[node]:
            if not visited[neighbor] and dist[neighbor] != -1:
                visited[neighbor] = True
                queue.append(neighbor)
    
    print(subtree_size)