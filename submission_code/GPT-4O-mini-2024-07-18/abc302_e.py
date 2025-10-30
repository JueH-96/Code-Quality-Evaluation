# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    first_line = data[0].split()
    N = int(first_line[0])
    Q = int(first_line[1])
    
    edges = [0] * (N + 1)  # To count edges for each vertex
    isolated_count = N  # Initially, all vertices are isolated
    results = []
    
    for i in range(1, Q + 1):
        query = list(map(int, data[i].split()))
        
        if query[0] == 1:  # Connect vertex u and vertex v
            u, v = query[1], query[2]
            if edges[u] == 0:
                isolated_count -= 1
            if edges[v] == 0:
                isolated_count -= 1
            
            edges[u] += 1
            edges[v] += 1
            
            if edges[u] == 1:
                isolated_count += 1
            if edges[v] == 1:
                isolated_count += 1
        
        elif query[0] == 2:  # Remove all edges connected to vertex v
            v = query[1]
            if edges[v] > 0:
                isolated_count -= 1  # v will become isolated
                edges[v] = 0  # Remove all edges from v
                # We need to check all vertices to see if they were connected to v
                for j in range(1, N + 1):
                    if j != v and edges[j] > 0:
                        edges[j] -= 1
                        if edges[j] == 0:
                            isolated_count += 1
        
        results.append(isolated_count)
    
    print("
".join(map(str, results)))

if __name__ == "__main__":
    main()