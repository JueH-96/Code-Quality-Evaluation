def solve():
    N, M = map(int, input().split())
    
    # Create adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Special case: if N=1, any number works
    if N == 1:
        print("Yes")
        print("1")
        return
        
    # For each vertex with degree >= 1, XOR of adjacent vertices must be 0
    # This means all adjacent vertices must have same value
    
    # Check if any vertex has exactly one neighbor
    for i in range(1, N+1):
        if len(adj[i]) == 1:
            print("No")
            return
            
    # If we reach here and have edges, we can assign same power of 2 to all vertices
    if M > 0:
        print("Yes")
        print("4 " * N)
    else:
        # No edges - any number works
        print("Yes") 
        print("1 " * N)

solve()