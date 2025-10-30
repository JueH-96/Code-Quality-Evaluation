def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    edges = input_data[2:2+2*(N-1)]
    specials = input_data[2+2*(N-1):2+2*(N-1)+K]

    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        a = int(edges[idx])
        b = int(edges[idx+1])
        idx += 2
        adj[a].append(b)
        adj[b].append(a)

    # Mark special nodes
    special = [0]*(N+1)
    for v in specials:
        special[int(v)] = 1

    # Prepare an array to store the number of special nodes in each subtree
    count_subtree = [0]*(N+1)

    # DFS to compute the size of subtree in terms of how many K-vertices it contains
    def dfs(u, p):
        c = special[u]
        for v in adj[u]:
            if v == p:
                continue
            c += dfs(v, u)
        count_subtree[u] = c
        return c

    # Run DFS from node 1 (any node works in a tree)
    dfs(1, -1)

    # The answer is the number of nodes that have count_subtree[u] > 0
    answer = sum(1 for i in range(1, N+1) if count_subtree[i] > 0)
    print(answer)

# Don't forget to call main()
if __name__ == "__main__":
    main()