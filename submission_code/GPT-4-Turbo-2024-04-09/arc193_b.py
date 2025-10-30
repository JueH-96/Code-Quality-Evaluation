def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    s = data[1]
    
    MOD = 998244353
    
    # Calculate the number of edges each vertex has in the undirected graph G
    # Vertex N has edges to all vertices i where s[i] == '1'
    count_ones = s.count('1')
    
    # Each vertex i (0 <= i < N) has:
    # - 1 edge to (i+1) % N
    # - 1 edge to (i-1) % N (or N-1 if i == 0)
    # - 1 edge to N if s[i] == '1'
    # Vertex N has count_ones edges to vertices where s[i] == '1'
    
    # Each edge can be directed in two ways, so for each edge, there are 2 choices.
    # Total edges in G:
    # - N edges for the cycle (0-1, 1-2, ..., N-1-0)
    # - count_ones edges from vertices i to N where s[i] == '1'
    total_edges = N + count_ones
    
    # Each edge can be directed in two ways independently
    # Total distinct directed graphs:
    total_directed_graphs = pow(2, total_edges, MOD)
    
    print(total_directed_graphs)

if __name__ == "__main__":
    main()