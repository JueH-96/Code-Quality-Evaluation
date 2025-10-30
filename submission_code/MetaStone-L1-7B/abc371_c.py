import sys
from itertools import permutations

def main():
    n = int(sys.stdin.readline())
    m_g = int(sys.stdin.readline())
    edges_g = []
    for _ in range(m_g):
        u, v = map(int, sys.stdin.readline().split())
        if u > v:
            u, v = v, u
        edges_g.append((u, v))
    e_g = set(edges_g)

    m_h = int(sys.stdin.readline())
    edges_h = []
    for _ in range(m_h):
        u, v = map(int, sys.stdin.readline().split())
        if u > v:
            u, v = v, u
        edges_h.append((u, v))
    e_h = set(edges_h)

    a = [[0] * n for _ in range(n)]
    for i in range(n - 1):
        line = list(map(int, sys.stdin.readline().split()))
        for j in range(i + 1, n):
            idx = j - (i + 1)
            a[i][j] = line[idx]
            a[j][i] = line[idx]

    min_cost = float('inf')

    for perm in permutations(range(n)):
        e_h_perm = set()
        for u, v in edges_h:
            new_u = perm[u]
            new_v = perm[v]
            if new_u > new_v:
                new_u, new_v = new_v, new_u
            e_h_perm.add((new_u, new_v))
        diff = e_g.symmetric_difference(e_h_perm)
        cost = sum(a[u][v] for u, v in diff)
        if cost < min_cost:
            min_cost = cost

    print(min_cost)

if __name__ == "__main__":
    main()