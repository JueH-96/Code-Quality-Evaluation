def main():
    N, Q = map(int, input().strip().split())
    
    # Initialize adjacency list for each vertex
    adj = [set() for _ in range(N+1)]  # 1-indexed
    
    # Initially all vertices are isolated
    isolated = set(range(1, N+1))
    
    for _ in range(Q):
        query = list(map(int, input().strip().split()))
        
        if query[0] == 1:  # Add edge
            u, v = query[1], query[2]
            
            # Add edge between u and v
            adj[u].add(v)
            adj[v].add(u)
            
            # u and v are no longer isolated
            isolated.discard(u)
            isolated.discard(v)
                
        else:  # Remove edges
            v = query[1]
            
            # Check all neighbors of v
            for u in list(adj[v]):
                adj[u].remove(v)
                # If a neighbor becomes isolated after removing the edge, add it to isolated set
                if not adj[u]:
                    isolated.add(u)
            
            # Clear all edges of v
            adj[v].clear()
            # v is now isolated
            isolated.add(v)
        
        # Print the number of isolated vertices
        print(len(isolated))

if __name__ == "__main__":
    main()