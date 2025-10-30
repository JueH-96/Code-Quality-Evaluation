import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    
    # Initialize adjacency list
    adj = defaultdict(set)
    # Initialize degree count
    degree = [0] * (N + 1)
    # Initialize the count of isolated vertices
    isolated = N
    
    for _ in range(Q):
        query_type = int(data[idx])
        if query_type == 1:
            u = int(data[idx+1])
            v = int(data[idx+2])
            idx += 3
            # Before adding the edge, check if u and v are isolated
            if degree[u] == 0:
                isolated -= 1
            if degree[v] == 0:
                isolated -= 1
            # Add the edge
            adj[u].add(v)
            adj[v].add(u)
            degree[u] += 1
            degree[v] += 1
        else:
            v = int(data[idx+1])
            idx += 2
            # Remove all edges connected to v
            for u in adj[v]:
                adj[u].remove(v)
                degree[u] -= 1
                if degree[u] == 0:
                    isolated += 1
            # Remove v's adjacency list
            adj[v].clear()
            degree[v] = 0
            if degree[v] == 0:
                isolated += 1
        # After processing the query, print the number of isolated vertices
        print(isolated)

if __name__ == "__main__":
    main()