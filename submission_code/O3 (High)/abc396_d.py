import sys
sys.setrecursionlimit(1 << 25)

# size of weights: 0 <= w < 2^60
MAX_BIT = 60   # we need bits 0 .. 59


def insert_basis(basis, x):
    """
    Insert number x into linear basis (Gaussian elimination on XOR space).
    `basis` is an array of length MAX_BIT where basis[i] has the highest set
    bit exactly at position i (or 0 if that position is empty).
    """
    y = x
    for b in range(MAX_BIT - 1, -1, -1):
        if not (y >> b) & 1:
            continue
        if basis[b] == 0:
            basis[b] = y
            return
        y ^= basis[b]


def minimise_with_basis(basis, value):
    """
    Given a linear basis `basis`, return the minimum number obtainable
    from `value` by XOR-ing any combination of basis vectors.
    """
    res = value
    for b in range(MAX_BIT - 1, -1, -1):
        if basis[b] and (res ^ basis[b]) < res:
            res ^= basis[b]
    return res


def main():
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    # adjacency list for DFS/BFS
    adj = [[] for _ in range(N)]
    edges = []

    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        adj[u].append((v, w))
        adj[v].append((u, w))
        edges.append((u, v, w))

    # 1. Build arbitrary spanning tree rooted at 0 (vertex 1)
    dist = [-1] * N          # XOR distance from vertex 1 along the tree
    dist[0] = 0
    stack = [0]
    while stack:
        v = stack.pop()
        for nv, w in adj[v]:
            if dist[nv] == -1:
                dist[nv] = dist[v] ^ w
                stack.append(nv)

    # 2. Build linear basis from fundamental cycle XORs
    basis = [0] * MAX_BIT
    for u, v, w in edges:
        cycle_xor = dist[u] ^ dist[v] ^ w
        if cycle_xor:
            insert_basis(basis, cycle_xor)

    # 3. Starting value is XOR along the tree path to N (vertex N)
    initial = dist[N - 1]
    answer = minimise_with_basis(basis, initial)

    print(answer)


if __name__ == "__main__":
    main()