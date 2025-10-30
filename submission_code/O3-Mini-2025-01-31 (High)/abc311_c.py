def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    # Create an array A where A[i] is the target vertex for i (using 1-indexing)
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = int(data[i])
    
    # We'll keep a global "visited" to ensure we process each vertex only once 
    visited = [False] * (N + 1)
    
    # For each vertex not visited, simulate traversing the graph following its edge.
    for i in range(1, N + 1):
        if not visited[i]:
            # Use a dictionary to record the index in our chain where each vertex was seen.
            pos = {}
            chain = []
            current = i
            while True:
                # If we've seen 'current' in the current chain, we've detected a cycle.
                if current in pos:
                    cycle = chain[pos[current]:]  # extract the cycle (vertices are unique)
                    sys.stdout.write(str(len(cycle)) + "
")
                    sys.stdout.write(" ".join(map(str, cycle)))
                    return
                # If current is already visited globally (but not in our chain),
                # then this chain will eventually merge with a component already processed.
                if visited[current]:
                    break
                pos[current] = len(chain)
                chain.append(current)
                visited[current] = True
                current = A[current]  # move to the next vertex following the edge
    
    # Given the constraints, a cycle is guaranteed to exist.

if __name__ == '__main__':
    main()