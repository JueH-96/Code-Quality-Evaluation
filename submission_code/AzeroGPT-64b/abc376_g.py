from itertools import accumulate
from collections import defaultdict

def inv(n, m):
    return pow(n, m - 2, m)

if __name__ == '__main__':
    m = 998244353
    kmod = [(inv(i, m) if i > 0 else 1) for i in range(1000000)]
    def invsum(x):
        s1, s2 = 0, 0
        for xi in x:
            s1 += xi * kmod[xi]
            s2 += xi
        s2 = inv(s2, m)
        return s1 * s2 % m * kmod[len(x)] % m

    def dfsx(tup):
        state, edges, nodes, cur = tup
        if len(edges) == 1:
            return [nodes[edges[0]]]
        state[edges[-1]] = 1
        dt = dfsx((state, edges[:-1], nodes, nodes[edges[-1]]))
        state[edges[-1]] = 0
        return [nodes[edges[-1]]] + dt

    t = int(input())
    for _ in range(t):
        n = int(input())
        edges = [-1] + list(map(int, input().split()))
        nodes = list(map(int, input().split()))
        
        nodes = [0] + nodes
        for i in range(1, len(nodes)):
            nodes[i] = nodes[i] * invsum(nodes) % m
        state = [False] * len(edges)
        state[0] = 1
        ans = []
        for ei in dfsx((state, list(range(len(edges))-1), edges, 0)):
            if not state[ei]:
                state[ei] = 1
                nn = sum(nodes) + nodes[ei] * m - nodes[ei]
                for i, ei in enumerate(edges):
                    if not state[ei]:
                        nn -= nodes[ei]
                ans.append(nodes[ei])
                total = nodes[ei]
                for i in range(len(edges)):
                    ei = edges[i]
                    nj = nodes[ei] * m - nodes[ei] + (nn if state[ei] else nodes[ei])
                    nodes[ei] = nj * (nn * m - nn + nodes[ei]) % m * kmod[nj + nodes[ei]] % m
                    nn = nj
                    if state[ei]: total = nj
                nodes[ei] = total
        a = sum(ans) * invsum(ans) % m
        print((a * inv(sum(nodes) - a, m) % m) if sum(nodes) != a else 0)