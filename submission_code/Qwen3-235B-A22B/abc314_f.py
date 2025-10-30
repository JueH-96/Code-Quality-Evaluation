MOD = 998244353

import sys

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    edges = []
    for _ in range(N-1):
        p = int(input[ptr]) - 1
        q = int(input[ptr+1]) - 1
        edges.append((p, q))
        ptr += 2

    parent = list(range(N))
    size = [1] * N
    score = [0] * N

    def find(u):
        if parent[u] != u:
            p = parent[u]
            parent[u] = find(parent[u])
            score[u] = (score[u] + score[p]) % MOD
        return parent[u]

    for u, v in edges:
        root_u = find(u)
        root_v = find(v)
        if root_u == root_v:
            continue
        su = size[root_u]
        sv = size[root_v]
        total = su + sv
        inv = pow(total, MOD-2, MOD)
        cu = su * inv % MOD
        cv = sv * inv % MOD

        if su < sv:
            root_u, root_v = root_v, root_u
            cu, cv = cv, cu

        parent[root_v] = root_u
        score[root_v] = (score[root_v] + cu) % MOD
        score[root_u] = (score[root_u] + cv) % MOD
        size[root_u] += size[root_v]

    # Final find to propagate all scores
    for i in range(N):
        find(i)
    
    print(' '.join(map(str, score)))

if __name__ == "__main__":
    main()