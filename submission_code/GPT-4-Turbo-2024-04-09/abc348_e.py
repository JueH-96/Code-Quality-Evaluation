import sys
input = sys.stdin.read
from collections import defaultdict, deque

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    
    edges = []
    for _ in range(N - 1):
        A = int(data[index]) - 1
        B = int(data[index + 1]) - 1
        edges.append((A, B))
        index += 2
    
    C = list(map(int, data[index:index + N]))
    
    # Build adjacency list for the tree
    tree = defaultdict(list)
    for A, B in edges:
        tree[A].append(B)
        tree[B].append(A)
    
    # Function to calculate distances from a given start node using BFS
    def bfs_distances(start):
        distances = [-1] * N
        distances[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            current_distance = distances[node]
            for neighbor in tree[node]:
                if distances[neighbor] == -1:  # not visited
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        return distances
    
    # Calculate f(x) for all x using BFS from any node, here we start from node 0
    initial_distances = bfs_distances(0)
    
    # Calculate initial f(0)
    f = [0] * N
    for i in range(N):
        f[0] += C[i] * initial_distances[i]
    
    # To find f(v) for all v, we can use the fact that moving the root from u to v changes the distance:
    # d(v, i) = d(u, i) - 1 if i is in the subtree of v when rooted at u
    # d(v, i) = d(u, i) + 1 otherwise
    # We need to efficiently calculate f(v) for all v using f(u) without recomputing from scratch
    
    # We can use a DFS to compute f(v) for all v based on f(u)
    def compute_all_f():
        stack = [(0, -1)]  # (current_node, parent_node)
        while stack:
            node, parent = stack.pop()
            for neighbor in tree[node]:
                if neighbor == parent:
                    continue
                # Calculate f(neighbor) using f(node)
                # f(neighbor) = f(node) + (sum(C) - 2 * C[neighbor])
                f[neighbor] = f[node] + (sum(C) - 2 * C[neighbor])
                stack.append((neighbor, node))
    
    # Compute f(v) for all v
    compute_all_f()
    
    # Find the minimum f(v)
    result = min(f)
    
    # Output the result
    print(result)

main()