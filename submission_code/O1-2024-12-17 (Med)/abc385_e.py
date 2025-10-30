def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    # Adjacency list
    adj = [[] for _ in range(N+1)]
    idx = 1
    for _ in range(N-1):
        u = int(input_data[idx]); v = int(input_data[idx+1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)

    # Degrees
    deg = [0]*(N+1)
    for i in range(1, N+1):
        deg[i] = len(adj[i])

    # We will try each node c as the "center".
    # For each c, gather L = [deg(v)-1 for v in adj[c]] (since v must connect to c and possibly keep its other neighbors as leaves).
    # Sort L descending, then for x in [1..len(L)], define y = min(L[0..x-1]) = L[x-1] (because sorted descending).
    # If y >= 1, the sub-snowflake uses = 1 (center) + x (arms) + x*y (leaves).
    # Track the maximum "uses" across all c.

    best_subtree_size = 0
    for c in range(1, N+1):
        if deg[c] < 1:
            # c is an isolated node if deg[c]=0 (shouldn't happen in a tree with N>1)
            # but just in case, skip
            continue
        # Build L array
        L = [deg[v] - 1 for v in adj[c]]  # deg[v]-1 possible leaves from neighbor v
        # Sort descending
        L.sort(reverse=True)
        min_so_far = float('inf')
        # Iterate x = number of arms
        # L[x-1] is the x-th largest value
        for x in range(1, len(L)+1):
            if L[x-1] < min_so_far:
                min_so_far = L[x-1]
            y = min_so_far
            if y < 1:
                # No point continuing if y < 1
                break
            used = 1 + x + x*y  # center + x arms + x*y leaves
            if used > best_subtree_size:
                best_subtree_size = used

    # The answer is how many vertices we delete = N - size_of_largest_snowflake_subtree
    print(N - best_subtree_size)

# Do not forget to call main!
if __name__ == "__main__":
    main()