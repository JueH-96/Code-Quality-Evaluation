import sys
sys.setrecursionlimit(10**7)

def main():
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:2*(N-1)+1]
    C = input_data[2*(N-1)+1:]
    C = list(map(int, C))

    if N == 1:
        # If there's only one vertex, the distances are all zero.
        print(0)
        return

    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        A = int(edges[idx])
        B = int(edges[idx+1])
        idx += 2
        adj[A].append(B)
        adj[B].append(A)

    # We'll root the tree at node 1.
    # subtree_sum[u] will store sum of C in the subtree rooted at u (including u).
    subtree_sum = [0]*(N+1)
    # We also need to compute f(1) = sum_{i} (C[i]*dist(1,i)).
    # We'll do a DFS passing the depth to accumulate that distance sum.
    total_dist = 0       # This will store the sum of C[i]*distance(1,i)
    totalC = sum(C)      # sum of all C[i]

    # First DFS: compute subtree sums + total_dist for root=1
    def dfs_build(u, p, depth):
        nonlocal total_dist
        subtree_sum[u] = C[u-1]  # C is 0-based, vertices are 1-based
        total_dist += C[u-1]*depth
        for v in adj[u]:
            if v == p:
                continue
            dfs_build(v, u, depth+1)
            subtree_sum[u] += subtree_sum[v]

    # Second DFS: re-root technique
    # If we have f(u) = sum of C[i]*dist(u, i),
    # then f(v) = f(u) + (totalC - 2*subtree_sum[v]) when v is a child of u.
    best = float('inf')
    def dfs_reroot(u, p, f_u):
        nonlocal best
        best = min(best, f_u)
        for v in adj[u]:
            if v == p:
                continue
            f_v = f_u + (totalC - 2*subtree_sum[v])
            dfs_reroot(v, u, f_v)

    # Run the first DFS from node=1
    dfs_build(1, -1, 0)
    # Now total_dist holds f(1)
    # Run the second DFS to find the minimum f(v)
    dfs_reroot(1, -1, total_dist)

    print(best)

# Do not forget to call main
if __name__ == "__main__":
    main()