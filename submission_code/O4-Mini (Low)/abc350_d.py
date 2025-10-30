import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    parent = list(range(N+1))
    comp_size = [1] * (N+1)
    edge_count = [0] * (N+1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            # already in same component, just record the extra edge
            edge_count[rx] += 1
        else:
            # union by size
            if comp_size[rx] < comp_size[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            comp_size[rx] += comp_size[ry]
            edge_count[rx] += edge_count[ry] + 1

    # process edges
    for _ in range(M):
        a, b = map(int, input().split())
        union(a, b)

    # accumulate answer: for each distinct root, final clique edges minus existing
    ans = 0
    seen = set()
    for i in range(1, N+1):
        r = find(i)
        if r not in seen:
            seen.add(r)
            k = comp_size[r]
            total_possible = k * (k - 1) // 2
            ans += total_possible - edge_count[r]

    print(ans)

if __name__ == "__main__":
    main()