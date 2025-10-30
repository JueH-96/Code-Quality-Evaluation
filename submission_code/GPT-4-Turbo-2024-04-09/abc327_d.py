def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    B = list(map(int, data[2+M:2+2*M]))
    
    from collections import defaultdict
    
    # Graph adjacency list
    adj = defaultdict(list)
    
    # Build the graph where each edge is between A_i and B_i
    for i in range(M):
        a = A[i]
        b = B[i]
        adj[a].append(b)
        adj[b].append(a)
    
    # To keep track of the colors assigned to each node
    color = [-1] * (N + 1)  # Using 1-based index
    
    def is_bipartite(v, c):
        # DFS function to check if graph is bipartite
        if color[v] != -1:
            return color[v] == c
        color[v] = c
        for neighbor in adj[v]:
            if not is_bipartite(neighbor, 1 - c):
                return False
        return True
    
    # Check each component
    for node in range(1, N + 1):
        if node in adj and color[node] == -1:
            if not is_bipartite(node, 0):
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()