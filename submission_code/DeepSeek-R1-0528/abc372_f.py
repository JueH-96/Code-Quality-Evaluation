MOD = 998244353

import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); K = int(next(it))
    extra_edges = []
    out_edges = defaultdict(list)
    vertices_with_extra = set()
    for _ in range(m):
        x = int(next(it)); y = int(next(it))
        extra_edges.append((x, y))
        out_edges[x].append(y)
        vertices_with_extra.add(x)
    S = sorted(vertices_with_extra)
    S_set = set(S)
    idx_in_S = {}
    for idx, s in enumerate(S):
        idx_in_S[s] = idx

    def next_vertex(v):
        if v < n:
            return v + 1
        else:
            return 1

    if K == 0:
        print(1)
        return

    F_prev = defaultdict(int)
    F_prev[1] = 1

    for step in range(1, K + 1):
        F_shift = defaultdict(int)
        for v, count_val in F_prev.items():
            nv = next_vertex(v)
            F_shift[nv] = (F_shift.get(nv, 0) + count_val) % MOD

        G = [0] * len(S)
        for j, s in enumerate(S):
            total = 0
            for y in out_edges[s]:
                total = (total + F_prev.get(y, 0)) % MOD
            G[j] = total

        F_current = F_shift
        for j, s in enumerate(S):
            if G[j] != 0:
                F_current[s] = (F_current.get(s, 0) + G[j]) % MOD

        if step == K:
            total_ans = 0
            for count_val in F_current.values():
                total_ans = (total_ans + count_val) % MOD
            print(total_ans)
            return

        F_prev = F_current

    total_ans = 0
    for count_val in F_prev.values():
        total_ans = (total_ans + count_val) % MOD
    print(total_ans)

if __name__ == '__main__':
    main()