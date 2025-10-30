import sys
from collections import defaultdict

def main():
    MOD = 998244353
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    pairs = []
    idx = 1
    for _ in range(N-1):
        a = int(data[idx])
        b = int(data[idx+1])
        pairs.append((a, b))
        idx += 2

    parent = list(range(N+1))  # players are 1-based
    size = [1] * (N + 1)
    sum_E = [0] * (N + 1)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for a, b in pairs:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            sum_E[ra] = (sum_E[ra] + sum_E[rb] + (size[ra] - size[rb]) * size[rb] // (size[ra] + size[rb])) % MOD

    result = [0] * (N + 1)
    for i in range(1, N+1):
        result[i] = sum_E[find(i)] % MOD

    print(' '.join(map(str, result[1:N+1])))

if __name__ == "__main__":
    main()