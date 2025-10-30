import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, D = map(int, input().split())
    coords = [tuple(map(int, input().split())) for _ in range(N)]
    D2 = D * D

    # Union-Find (Disjoint Set Union) implementation
    parent = list(range(N))
    rank = [0] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        else:
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

    # Build connections for pairs within distance D
    for i in range(N):
        xi, yi = coords[i]
        for j in range(i + 1, N):
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            if dx*dx + dy*dy <= D2:
                union(i, j)

    root0 = find(0)
    out = []
    for i in range(N):
        out.append("Yes" if find(i) == root0 else "No")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()