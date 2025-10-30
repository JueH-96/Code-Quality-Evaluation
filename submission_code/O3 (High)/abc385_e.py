import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))

    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
        adj[v].append(u)

    degree = [len(nei) for nei in adj]

    max_keep = 0                              # size of the largest snowflake we can keep
    for root in range(n):
        # For every neighbour, how many other neighbours (potential leaves) does it have?
        candidates = [degree[nb] - 1 for nb in adj[root] if degree[nb] > 1]  # keep only if >= 1
        if not candidates:
            continue

        candidates.sort(reverse=True)         # descending order

        for idx, val in enumerate(candidates, 1):   # idx is 1 … len(candidates)
            # minimal deg2 in the prefix of length idx is `val`
            size = 1 + idx * (1 + val)        # 1(root) + idx spines + idx*val leaves
            if size > max_keep:
                max_keep = size

    # The problem statement guarantees at least one feasible snowflake, so max_keep > 0
    print(n - max_keep)


if __name__ == "__main__":
    main()