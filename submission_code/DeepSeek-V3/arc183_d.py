import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A, B = map(int, sys.stdin.readline().split())
        edges[A].append(B)
        edges[B].append(A)
    
    # Initialize degrees
    degree = [0] * (N+1)
    for i in range(1, N+1):
        degree[i] = len(edges[i])
    
    # Initialize queue with leaves
    q = deque()
    for i in range(1, N+1):
        if degree[i] == 1:
            q.append(i)
    
    # To store the order of removal
    order = []
    while q:
        u = q.popleft()
        if degree[u] == 0:
            continue
        # Find the neighbor
        for v in edges[u]:
            if degree[v] > 0:
                break
        # Remove u and v
        order.append((u, v))
        degree[u] = 0
        degree[v] = 0
        # Update degrees of neighbors of v
        for neighbor in edges[v]:
            if degree[neighbor] > 0:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    q.append(neighbor)
    
    # Now, calculate the distance for each pair
    # Since the tree is a line, the distance between u and v is the number of edges between them
    # But to maximize the total score, we need to pair the farthest leaves first
    # So, we need to find the diameter of the tree and pair the leaves accordingly
    
    # To find the diameter, perform BFS twice
    def bfs(start):
        visited = [-1] * (N+1)
        q = deque()
        q.append(start)
        visited[start] = 0
        farthest = start
        max_dist = 0
        while q:
            u = q.popleft()
            for v in edges[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    q.append(v)
                    if visited[v] > max_dist:
                        max_dist = visited[v]
                        farthest = v
        return farthest, max_dist
    
    # Find one end of the diameter
    start = 1
    farthest, _ = bfs(start)
    # Find the other end
    farthest2, diameter = bfs(farthest)
    
    # Now, we need to pair the leaves in such a way that the sum of distances is maximized
    # One way is to pair the leaves in the order of their distance from one end of the diameter
    # So, we can perform a BFS from one end and pair the leaves in the order of their distance
    
    # Perform BFS from farthest
    visited = [-1] * (N+1)
    q = deque()
    q.append(farthest)
    visited[farthest] = 0
    while q:
        u = q.popleft()
        for v in edges[u]:
            if visited[v] == -1:
                visited[v] = visited[u] + 1
                q.append(v)
    
    # Now, sort the leaves based on their distance from farthest
    leaves = []
    for i in range(1, N+1):
        if degree[i] == 1:
            leaves.append(i)
    leaves.sort(key=lambda x: visited[x], reverse=True)
    
    # Pair the leaves in order
    paired = [False] * (N+1)
    result = []
    for i in range(0, len(leaves), 2):
        u = leaves[i]
        v = leaves[i+1]
        result.append((u, v))
        paired[u] = True
        paired[v] = True
    
    # Now, print the result
    for pair in result:
        print(pair[0], pair[1])

if __name__ == "__main__":
    main()