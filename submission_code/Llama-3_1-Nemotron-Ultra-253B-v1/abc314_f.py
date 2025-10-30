import sys
MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    parent = list(range(N+1))
    size = [1] * (N+1)
    weight = [0] * (N+1)
    
    def find(u):
        if parent[u] == u:
            return u
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        # u is root
        # Process path in reverse
        cum_sum = 0
        for node in reversed(path):
            cum_sum += weight[node]
            weight[node] = cum_sum % MOD
            parent[node] = u
        return u
    
    for _ in range(N-1):
        p, q = map(int, sys.stdin.readline().split())
        root_p = find(p)
        root_q = find(q)
        if root_p == root_q:
            continue
        a = size[root_p]
        b = size[root_q]
        den = a + b
        inv_den = pow(den, MOD-2, MOD)
        prob_a = (a * inv_den) % MOD
        prob_b = (b * inv_den) % MOD
        # Add probabilities to the roots
        weight[root_p] = (weight[root_p] + prob_a) % MOD
        weight[root_q] = (weight[root_q] + prob_b) % MOD
        # Merge smaller into larger
        if a > b:
            root_p, root_q = root_q, root_p
        parent[root_p] = root_q
        size[root_q] += size[root_p]
    
    # Compute results
    result = []
    for i in range(1, N+1):
        find(i)  # Ensure path is compressed
        result.append(weight[i] % MOD)
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()