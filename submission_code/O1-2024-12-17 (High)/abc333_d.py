def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]

    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        u = int(edges[idx])
        v = int(edges[idx+1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)

    # Array to store subtree sizes when rooted at 1
    subtree_size = [0]*(N+1)

    # DFS to compute subtree sizes
    def dfs(u, parent):
        subtree_size[u] = 1
        for w in adj[u]:
            if w != parent:
                dfs(w, u)
                subtree_size[u] += subtree_size[w]

    # Run DFS from vertex 1
    dfs(1, -1)

    # If vertex 1 has no neighbors, it would be isolated, which can't happen here as N >= 2 and it's a tree.
    # Otherwise, find the largest subtree size among 1's neighbors.
    if not adj[1]:
        # If for some reason it had no neighbors (deg(1) == 0), the answer would trivially be 1
        print(1)
        return

    # The minimum number of operations required to delete vertex 1
    # is N - max(subtree_size[child]) for child in adj[1].
    # (Because removing those other subtrees forces 1 to become a leaf, then remove 1.)
    max_subtree = max(subtree_size[child] for child in adj[1])
    answer = N - max_subtree
    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()