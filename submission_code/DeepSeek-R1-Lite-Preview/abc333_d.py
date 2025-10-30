def main():
    import sys
    from collections import defaultdict, deque

    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    edges = N_and_rest[1:]
    adj = defaultdict(list)
    for i in range(0, 2*(N-1), 2):
        u = int(edges[i])
        v = int(edges[i+1])
        adj[u].append(v)
        adj[v].append(u)
    
    if N == 1:
        print(1)
        return
    
    # Check if vertex 1 is a leaf
    if len(adj[1]) == 1:
        print(1)
        return
    else:
        # Count the number of leaves
        leaves = 0
        for v in range(1, N+1):
            if len(adj[v]) == 1:
                leaves += 1
        print(leaves)

if __name__ == '__main__':
    main()