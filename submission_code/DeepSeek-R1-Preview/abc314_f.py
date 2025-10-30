import sys
MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    matches = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N-1)]
    
    max_inv = 4 * 10**5
    inv = [1] * (max_inv + 2)
    for i in range(2, max_inv + 1):
        inv[i] = MOD - MOD // i * inv[MOD % i] % MOD

    parent = list(range(N + 1))
    size = [1] * (N + 1)
    add = [0] * (N + 1)

    def find(u):
        if parent[u] != u:
            orig_parent = parent[u]
            parent[u] = find(parent[u])
            add[u] += add[orig_parent]
            add[u] %= MOD
        return parent[u]

    for p, q in matches:
        root_p = find(p)
        root_q = find(q)
        a = size[root_p]
        b = size[root_q]
        sum_ab = a + b
        inv_sum_ab = inv[sum_ab]
        prob_p = a * inv_sum_ab % MOD
        prob_q = b * inv_sum_ab % MOD

        add[root_p] = (add[root_p] + prob_p) % MOD
        add[root_q] = (add[root_q] + prob_q) % MOD

        parent[root_q] = root_p
        add[root_q] = (add[root_q] - add[root_p]) % MOD
        size[root_p] += size[root_q]

    result = []
    for i in range(1, N + 1):
        find(i)
        res = add[i] % MOD
        result.append(str(res))
    print(' '.join(result))

if __name__ == '__main__':
    main()