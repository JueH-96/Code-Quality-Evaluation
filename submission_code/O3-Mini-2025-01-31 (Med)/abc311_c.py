def main():
    import sys
    sys.setrecursionlimit(300000)  # Increase recursion limit if needed
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # Parse the input
    n = int(input_data[0])
    A = [0] * (n + 1)
    for i in range(1, n + 1):
        A[i] = int(input_data[i])
    
    # We will use a visited list to mark global visited nodes
    visited = [False] * (n + 1)
    
    # For each vertex, if not visited, start following the chain
    for i in range(1, n + 1):
        if visited[i]:
            continue
        
        # We keep a dictionary to remember the index in the chain when a vertex is first seen.
        chain_index = {}
        chain = []
        
        # Start following the chain beginning at vertex i.
        u = i
        while True:
            # If we see u again in the current chain, we have found a cycle.
            if u in chain_index:
                cycle_start = chain_index[u]
                cycle = chain[cycle_start:]
                # Output the cycle (its length and vertices)
                print(len(cycle))
                print(" ".join(map(str, cycle)))
                return
            
            # If u was seen in a previous chain, then no cycle will be found from this branch.
            if visited[u]:
                break
            
            # Mark the vertex in the current chain and update global visited.
            chain_index[u] = len(chain)
            chain.append(u)
            visited[u] = True
            
            # Move to the next vertex using the directed edge u -> A[u]
            u = A[u]
            
    # Since the constraints guarantee a cycle exists, we should never reach here.
    
if __name__ == '__main__':
    main()