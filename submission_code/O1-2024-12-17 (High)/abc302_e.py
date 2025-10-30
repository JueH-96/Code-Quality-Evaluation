def main():
    import sys
    input_data = sys.stdin.read().split()
    
    N, Q = map(int, input_data[:2])
    idx = 2
    
    # Adjacency list (using sets for efficient edge removal),
    # degrees array to count the number of edges per vertex.
    adjacency = [set() for _ in range(N+1)]
    degrees = [0] * (N+1)
    
    # Initially all vertices have degree 0 => all are "isolated".
    zero_count = N
    
    # We'll collect the output in a list for efficient printing at the end.
    answers = []
    
    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        
        if t == 1:
            # "1 u v" => connect u and v with an edge
            u = int(input_data[idx]); idx += 1
            v = int(input_data[idx]); idx += 1
            
            # Add edge to adjacency set
            adjacency[u].add(v)
            adjacency[v].add(u)
            
            # Update degrees - if it was 0, it no longer is
            if degrees[u] == 0:
                zero_count -= 1
            degrees[u] += 1
            if degrees[v] == 0:
                zero_count -= 1
            degrees[v] += 1
            
            answers.append(str(zero_count))
        
        else:
            # "2 v" => remove all edges from v
            v = int(input_data[idx]); idx += 1
            
            if degrees[v] > 0:
                # For each neighbor, remove v from that neighbor's adjacency
                for w in adjacency[v]:
                    adjacency[w].remove(v)
                    degrees[w] -= 1
                    if degrees[w] == 0:
                        zero_count += 1
                
                # Clear v's adjacency
                adjacency[v].clear()
                
                # v becomes isolated if it wasn't already (degree was > 0)
                degrees[v] = 0
                zero_count += 1
            
            answers.append(str(zero_count))
    
    print("
".join(answers))

# Do not forget to call main()
if __name__ == "__main__":
    main()