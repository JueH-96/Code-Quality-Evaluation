import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    size_tree = 1
    while size_tree < N:
        size_tree <<= 1
    return_time = [0] * (N + 1)
    tree = [float('inf')] * (2 * size_tree)
    for i in range(1, N + 1):
        tree[size_tree + i - 1] = return_time[i]
    for i in range(size_tree + N, 2 * size_tree):
        tree[i] = float('inf')
    for i in range(size_tree - 1, 0, -1):
        tree[i] = min(tree[2 * i], tree[2 * i + 1])
    totals = [0] * (N + 1)

    for _ in range(M):
        T_i, W_i, S_i = map(int, sys.stdin.readline().split())
        res = None
        def _query(node, l, r):
            nonlocal res
            if tree[node] > T_i:
                return
            if l == r:
                if l <= N and (res is None or l < res):
                    res = l
                return
            mid = (l + r) // 2
            _query(2 * node, l, mid)
            if res is not None:
                return
            _query(2 * node + 1, mid + 1, r)
        _query(1, 1, size_tree)
        if res is not None:
            i = res
            totals[i] += W_i
            new_time = T_i + S_i
            pos = size_tree + i - 1
            tree[pos] = new_time
            pos >>= 1
            while pos >= 1:
                new_val = min(tree[2 * pos], tree[2 * pos + 1])
                if tree[pos] == new_val:
                    break
                tree[pos] = new_val
                pos >>= 1

    for i in range(1, N + 1):
        print(totals[i])

if __name__ == "__main__":
    main()