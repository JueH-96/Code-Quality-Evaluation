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
    # Initialize the count of isolated vertices
    isolated = N
    
    for _ in range(Q):
        type_query = int(data[idx])
        if type_query == 1:
            u = int(data[idx+1])
            v = int(data[idx+2])
            idx += 3
            # Before adding the edge, check if u and v are isolated
            if len(adj[u]) == 0:
                isolated -= 1
            if len(adj[v]) == 0:
                isolated -= 1
            # Add the edge
            adj[u].add(v)
            adj[v].add(u)
        else:
            v = int(data[idx+1])
            idx += 2
            # Remove all edges connected to v
            for u in adj[v]:
                adj[u].remove(v)
                if len(adj[u]) == 0:
                    isolated += 1
            # After removing all edges, check if v is now isolated
            if len(adj[v]) == 0:
                isolated += 1
            adj[v].clear()
        # Output the number of isolated vertices
        print(isolated)

if __name__ == "__main__":
    main()