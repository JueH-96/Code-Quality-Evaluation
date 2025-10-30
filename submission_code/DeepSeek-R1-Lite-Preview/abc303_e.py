def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import defaultdict, deque

    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = data[1:]
    adj = defaultdict(list)
    for i in range(0, len(edges), 2):
        u = int(edges[i])
        v = int(edges[i+1])
        adj[u].append(v)
        adj[v].append(u)
    
    degrees = [len(adj[i]) for i in range(1, N+1)]
    leaves = degrees.count(1)
    
    # Calculate M
    M = (N - leaves + 2) // 3
    
    # Find vertices with degree > 2
    high_degree_vertices = [d for d in degrees if d > 2]
    
    # Number of stars with k_i=2
    num_k2 = M - len(high_degree_vertices)
    
    # Collect all k_i's
    k_list = high_degree_vertices + [2] * num_k2
    
    # Sort and print
    k_list.sort()
    print(' '.join(map(str, k_list)))

if __name__ == '__main__':
    main()