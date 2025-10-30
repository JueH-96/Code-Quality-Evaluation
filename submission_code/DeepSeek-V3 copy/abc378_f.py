import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    # Find the diameter of the tree
    def bfs(start):
        visited = [-1] * (N+1)
        queue = deque()
        queue.append(start)
        visited[start] = 0
        farthest_node = start
        max_dist = 0
        while queue:
            u = queue.popleft()
            for v in edges[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    queue.append(v)
                    if visited[v] > max_dist:
                        max_dist = visited[v]
                        farthest_node = v
        return farthest_node, max_dist
    
    # Find the diameter
    start = 1
    farthest, _ = bfs(start)
    farthest2, diameter = bfs(farthest)
    
    # If diameter is not even, no solution
    if diameter % 2 != 0:
        print(0)
        return
    
    # Find the center(s)
    radius = diameter // 2
    # Find the path from farthest2 to farthest
    path = []
    visited = [-1] * (N+1)
    queue = deque()
    queue.append(farthest2)
    visited[farthest2] = 0
    parent = [0] * (N+1)
    while queue:
        u = queue.popleft()
        if u == farthest:
            break
        for v in edges[u]:
            if visited[v] == -1:
                visited[v] = visited[u] + 1
                parent[v] = u
                queue.append(v)
    # Reconstruct the path
    u = farthest
    while u != farthest2:
        path.append(u)
        u = parent[u]
    path.append(farthest2)
    path.reverse()
    
    # The center is the middle node(s)
    if diameter % 2 == 0:
        center = path[radius]
    else:
        center1 = path[radius]
        center2 = path[radius+1]
        # Since diameter is even, only one center
        center = center1
    
    # Now, we need to find all pairs of nodes that are at distance radius from the center
    # and are not connected directly
    # First, find all nodes at distance radius from the center
    visited = [-1] * (N+1)
    queue = deque()
    queue.append(center)
    visited[center] = 0
    candidates = []
    while queue:
        u = queue.popleft()
        if visited[u] == radius:
            candidates.append(u)
        for v in edges[u]:
            if visited[v] == -1:
                visited[v] = visited[u] + 1
                queue.append(v)
    
    # Now, for each candidate, check if it is connected to any other candidate
    # If not, then it can form a cycle with another candidate
    # We need to count the number of such pairs
    # Since the tree is a tree, the candidates are all leaves at distance radius from the center
    # and they are not connected to each other
    # So, the number of such pairs is the number of candidates choose 2
    # But we need to ensure that adding the edge between any two candidates forms a cycle where all vertices in the cycle have degree 3
    # Since the candidates are all at distance radius from the center, and the center is connected to all of them, adding an edge between any two candidates will form a cycle of length 2*radius + 1
    # For all vertices in the cycle to have degree 3, the center must have degree 2 (since it is connected to the two candidates and the rest of the tree)
    # So, the center must have degree 2
    # Check the degree of the center
    if len(edges[center]) != 2:
        print(0)
        return
    
    # Now, count the number of candidates
    k = len(candidates)
    # The number of valid pairs is k * (k - 1) // 2
    print(k * (k - 1) // 2)

if __name__ == "__main__":
    main()