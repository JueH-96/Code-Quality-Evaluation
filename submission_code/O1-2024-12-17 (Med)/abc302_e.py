def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Parse N and Q
    N, Q = map(int, input_data[:2])
    
    # Adjacency list (using set for efficient removal)
    adjacency = [set() for _ in range(N+1)]
    
    # Degree array
    deg = [0]*(N+1)
    
    # Count of isolated vertices
    iso_count = N
    
    # Pointer to read queries
    idx = 2
    
    output = []
    
    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        
        if t == 1:
            # 1 u v: add edge between u and v
            u = int(input_data[idx]); idx += 1
            v = int(input_data[idx]); idx += 1
            
            # Add to adjacency
            adjacency[u].add(v)
            adjacency[v].add(u)
            
            # Update degrees and iso_count
            if deg[u] == 0:
                iso_count -= 1
            if deg[v] == 0:
                iso_count -= 1
            deg[u] += 1
            deg[v] += 1
            
            output.append(str(iso_count))
            
        else:
            # 2 v: remove all edges from v
            v = int(input_data[idx]); idx += 1
            
            # Remove v from neighbors
            for x in adjacency[v]:
                adjacency[x].remove(v)
                deg[x] -= 1
                if deg[x] == 0:
                    iso_count += 1
            
            # If v had degree > 0, it becomes isolated
            if deg[v] > 0:
                deg[v] = 0
                iso_count += 1
            
            # Clear adjacency of v
            adjacency[v].clear()
            
            output.append(str(iso_count))
    
    print("
".join(output))

# Do not forget to call main
main()