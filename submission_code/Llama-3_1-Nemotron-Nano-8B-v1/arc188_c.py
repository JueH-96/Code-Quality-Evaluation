import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

def main():
    N, M = map(int, stdin.readline().split())
    edges = []
    for _ in range(M):
        a, b, c = map(int, stdin.readline().split())
        a -= 1
        b -= 1
        edges.append((a, b, c))

    parent = list(range(N))
    rank = [1] * N
    parity = [0] * N  # 0 or 1, represents the parity of the path from node to root

    def find(u):
        if parent[u] == u:
            return u, 0
        p, p_parity = find(parent[u])
        parity[u] ^= p_parity
        parent[u] = p
        return parent[u], parity[u]

    is_bipartite = True
    for a, b, c in edges:
        ra, pa = find(a)
        rb, pb = find(b)
        if ra != rb:
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
                a, b = b, a
                pa, pb = pb, pa
            parent[rb] = ra
            parity[rb] = pa ^ pb ^ c
            rank[ra] += rank[rb]
        else:
            if (pa ^ pb) != c:
                is_bipartite = False

    if is_bipartite:
        print('0' * N)
        return

    candidate = [-1] * N
    for i in range(N):
        parent = list(range(N))
        rank = [1] * N
        parity = [0] * N
        bipart = True
        for a, b, c in edges:
            if a == i or b == i:
                continue
            ra, pa = find(a)
            rb, pb = find(b)
            if ra != rb:
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                    a, b = b, a
                    pa, pb = pb, pa
                parent[rb] = ra
                parity[rb] = pa ^ pb ^ c
                rank[ra] += rank[rb]
            else:
                if (pa ^ pb) != c:
                    bipart = False
                    break
        if bipart:
            candidate[i] = 1

    s = sum(candidate)
    if s != -1:
        res = [0] * N
        for i in range(N):
            if candidate[i] == 1:
                res[i] = '1'
            else:
                res[i] = '0'
        print(''.join(res))
    else:
        print(-1)

if __name__ == '__main__':
    main()