import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    from collections import defaultdict

    # Using list of sets for adjacency list
    adj = [set() for _ in range(N + 2)]  # 1-based indexing
    degree = [0] * (N + 2)
    isolated_count = N

    for _ in range(Q):
        t = int(input[ptr])
        ptr += 1
        if t == 1:
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            # Add edge u-v
            for x in [u, v]:
                y = v if x == u else u
                prev_zero = (degree[x] == 0)
                adj[x].add(y)
                degree[x] += 1
                if prev_zero:
                    isolated_count -= 1
        elif t == 2:
            v = int(input[ptr])
            ptr += 1
            # Remove all edges connected to v
            neighbors = list(adj[v])
            for u in neighbors:
                # Remove v from u's adjacency list
                adj[u].discard(v)
                # Update u's degree and isolated_count
                if degree[u] > 0:
                    old_degree = degree[u]
                    degree[u] -= 1
                    if old_degree == 1:
                        isolated_count += 1
            # Clear v's adjacency list and update its degree
            adj[v].clear()
            prev_degree_v = degree[v]
            degree[v] = 0
            if prev_degree_v > 0:
                isolated_count += 1
        print(isolated_count)

if __name__ == "__main__":
    main()