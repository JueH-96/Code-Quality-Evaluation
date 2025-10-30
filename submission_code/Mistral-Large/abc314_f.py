import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    p = [0] * (N - 1)
    q = [0] * (N - 1)

    for i in range(N - 1):
        p[i] = int(data[2 * i + 1]) - 1
        q[i] = int(data[2 * i + 2]) - 1

    parent = list(range(N))
    size = [1] * N
    expected_wins = [0] * N

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            size_u = size[root_u]
            size_v = size[root_v]
            total_size = size_u + size_v
            prob_u_wins = size_u / total_size
            prob_v_wins = size_v / total_size

            for i in range(N):
                if find(i) == root_u:
                    expected_wins[i] += prob_u_wins
                elif find(i) == root_v:
                    expected_wins[i] += prob_v_wins

            parent[root_v] = root_u
            size[root_u] = total_size

    for i in range(N - 1):
        union(p[i], q[i])

    for i in range(N):
        expected_wins[i] = int(expected_wins[i] % MOD)

    print(" ".join(map(str, expected_wins)))

if __name__ == "__main__":
    main()