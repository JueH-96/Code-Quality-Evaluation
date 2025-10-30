def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]
    
    # Build adjacency list and degree array
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]); v = int(edges[idx+1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # We will decide for each vertex whether it is a 'center' or 'leaf'
    # according to the procedure described:
    #   1) Sort vertices in descending order by degree.
    #   2) If a vertex is not yet decided, mark it as a center.
    #   3) Mark all its undecided neighbors as leaves.
    # A node with degree >= 3 must be a center (since leaves can only end up
    # with degree <= 2).  In case of ties at degree=2, whichever we encounter
    # first in the sorted order becomes a center, its neighbors become leaves,
    # which is consistent with there being a unique valid decomposition.
    
    # None => not decided; True => center; False => leaf
    center_flag = [None]*(N+1)

    # Sort all vertices in descending order of degree
    nodes_sorted = sorted(range(1, N+1), key=lambda x: deg[x], reverse=True)

    for u in nodes_sorted:
        if center_flag[u] is None:
            # Mark u as a center
            center_flag[u] = True
            # Mark neighbors (if undecided) as leaves
            for w in adj[u]:
                if center_flag[w] is None:
                    center_flag[w] = False

    # Now collect the degrees of all centers; those degrees are the star levels
    levels = []
    for i in range(1, N+1):
        if center_flag[i]:
            levels.append(deg[i])

    # Sort levels in ascending order and print
    levels.sort()
    print(" ".join(map(str, levels)))

# Do not forget to call main()
if __name__ == "__main__":
    main()