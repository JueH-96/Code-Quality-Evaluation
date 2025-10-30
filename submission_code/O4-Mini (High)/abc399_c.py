import sys

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    parent = list(range(N+1))

    def find(x):
        # iterative path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = 0
    for _ in range(M):
        u, v = map(int, input().split())
        ru = find(u)
        rv = find(v)
        if ru != rv:
            parent[rv] = ru
            merges += 1

    # In each component a spanning tree has (size-1) edges,
    # total edges in a forest = N - (#components) = merges
    # Extra edges to remove = M - merges
    print(M - merges)

if __name__ == "__main__":
    main()