import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    # Find all vertices with degree 2
    degree = [0] * (N+1)
    for u in range(1, N+1):
        degree[u] = len(edges[u])
    
    # Find all vertices with degree 2
    candidates = [u for u in range(1, N+1) if degree[u] == 2]
    
    # Now, we need to find pairs of candidates that are not directly connected
    # and adding an edge between them forms a cycle where all vertices in the cycle have degree 3
    # To do this, we need to find all pairs of candidates that are not adjacent and are at least 2 steps apart
    
    # First, find the distance between all pairs of candidates
    # Since the tree is connected, we can use BFS to find the distance between any two nodes
    # However, for large N, this is not feasible. Instead, we need a smarter approach
    
    # Since the tree is a tree, the path between any two nodes is unique
    # We need to find pairs of candidates where the path between them has all internal nodes with degree 2
    # and the path length is at least 2
    
    # To find such pairs, we can perform a BFS from each candidate and find other candidates that are reachable via a path of degree 2 nodes
    
    # First, create a list of all candidates
    candidate_set = set(candidates)
    
    # Now, for each candidate, perform BFS to find other candidates reachable via a path of degree 2 nodes
    # and count the number of such pairs
    
    # To avoid double counting, we can process each candidate in order and only consider pairs where the second candidate is greater than the first
    
    count = 0
    for u in candidates:
        visited = {u}
        q = deque()
        q.append((u, 0))
        while q:
            current, dist = q.popleft()
            if dist >= 2 and current in candidate_set and current > u:
                count += 1
            for neighbor in edges[current]:
                if neighbor not in visited and degree[neighbor] == 2:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
    
    print(count)

if __name__ == "__main__":
    main()