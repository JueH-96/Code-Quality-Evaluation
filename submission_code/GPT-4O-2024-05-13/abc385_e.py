# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]
    
    if N == 3:
        print(0)
        return
    
    # Create adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find the node with the highest degree
    max_degree = 0
    central_node = -1
    for node in range(1, N+1):
        degree = len(adj[node])
        if degree > max_degree:
            max_degree = degree
            central_node = node
    
    # BFS to classify nodes
    queue = deque([central_node])
    visited = [False] * (N + 1)
    visited[central_node] = True
    layer1 = []
    layer2 = []
    
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                if node == central_node:
                    layer1.append(neighbor)
                else:
                    layer2.append(neighbor)
    
    # Check the degrees of nodes in layer1 and layer2
    deletions = 0
    for node in layer1:
        if len(adj[node]) != len(layer2) + 1:
            deletions += 1
    for node in layer2:
        if len(adj[node]) != 1:
            deletions += 1
    
    print(deletions)

if __name__ == "__main__":
    main()