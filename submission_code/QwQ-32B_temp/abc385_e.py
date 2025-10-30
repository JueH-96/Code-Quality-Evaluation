import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    degrees = [0]*(N+1)
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    # Precompute non_leaf_neighbors and count_non_leaf
    non_leaf_neighbors = [set() for _ in range(N+1)]
    for v in range(1, N+1):
        for w in adj[v]:
            if degrees[w] > 1:
                non_leaf_neighbors[v].add(w)
    count_non_leaf = [0]*(N+1)
    for v in range(1, N+1):
        count_non_leaf[v] = len(non_leaf_neighbors[v])

    max_val = 0
    for u in range(1, N+1):
        x = degrees[u]
        neighbors = adj[u]
        if not neighbors:
            continue
        # Check all neighbors have the same degree
        first_degree = degrees[neighbors[0]]
        valid = True
        for v in neighbors[1:]:
            if degrees[v] != first_degree:
                valid = False
                break
        if not valid:
            continue
        d = first_degree
        # Now check each neighbor v in neighbors
        for v in neighbors:
            cnt = count_non_leaf[v]
            if cnt not in (0, 1):
                valid = False
                break
            if cnt == 1:
                if u not in non_leaf_neighbors[v]:
                    valid = False
                    break
            else:  # cnt ==0
                if u in non_leaf_neighbors[v]:
                    valid = False
                    break
        if valid:
            current_val = 1 + x + x*(d - 1)
            if current_val > max_val:
                max_val = current_val
    print(N - max_val)

if __name__ == "__main__":
    main()