import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    parent = list(range(N + 1))
    rank = [1] * (N + 1)
    components = N

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            components -= 1
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v
                if rank[root_u] == rank[root_v]:
                    rank[root_v] += 1

    ans = M - (N - components)
    print(ans)

if __name__ == "__main__":
    main()