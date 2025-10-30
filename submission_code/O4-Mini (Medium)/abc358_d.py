import sys
import threading

def main():
    import sys
    import bisect

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Sort box candy/prices
    A.sort()
    # Sort requirements descending
    B.sort(reverse=True)

    # DSU for "next free index" in A
    parent = list(range(N + 1))

    def find(x):
        # path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        # make the representative of x point to rep of y
        rx = find(x)
        ry = find(y)
        parent[rx] = ry

    total = 0
    for b in B:
        # find smallest index in A with A[idx] >= b
        idx = bisect.bisect_left(A, b)
        if idx == N:
            print(-1)
            return
        # find the next free index >= idx
        free_idx = find(idx)
        if free_idx == N:
            # no free boxes left that satisfy this requirement
            print(-1)
            return
        # assign box at free_idx
        total += A[free_idx]
        # mark it used: union it with next
        union(free_idx, free_idx + 1)

    print(total)

if __name__ == "__main__":
    main()