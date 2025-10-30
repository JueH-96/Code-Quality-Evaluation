import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return

    # Build adjacency list and degree array
    adj = [[] for _ in range(N+1)]
    deg = [0] * (N+1)
    for _ in range(N-1):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    max_keep = 0

    # For each vertex r as potential center
    for r in range(1, N+1):
        # Build list of how many leaves each neighbor can supply: L_u = deg[u] - 1
        L = [deg[u] - 1 for u in adj[r]]
        if not L:
            continue
        # Sort in descending order so that the x-th best neighbor has L_x
        L.sort(reverse=True)

        best_cand = 0
        # Try x = 1 .. len(L), y = L[x-1], but only if y >= 1
        for i, y in enumerate(L):
            if y <= 0:
                break
            x = i + 1
            # Number of non-center vertices kept is x + x*y = x*(y+1)
            cand = x * (y + 1)
            if cand > best_cand:
                best_cand = cand

        if best_cand > 0:
            # Total kept is 1 (the center) + best_cand
            keep = 1 + best_cand
            if keep > max_keep:
                max_keep = keep

    # Answer = total vertices - maximum we can keep
    print(N - max_keep)

if __name__ == "__main__":
    main()