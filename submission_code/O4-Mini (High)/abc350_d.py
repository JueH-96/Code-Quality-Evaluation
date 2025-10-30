import sys

def main():
    input = sys.stdin.readline
    n, m = map(int, input().split())

    # Disjoint Set Union (union by size + path compression)
    parent = list(range(n + 1))
    size = [1] * (n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        # attach smaller tree to larger
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    # Read friendships and union them
    for _ in range(m):
        a, b = map(int, input().split())
        union(a, b)

    # For each component, we can end up with a clique of size s,
    # i.e. s*(s-1)/2 edges. Sum those up and subtract the
    # already-existing m edges.
    total_possible = 0
    for i in range(1, n + 1):
        if parent[i] == i:   # i is a root
            s = size[i]
            total_possible += s * (s - 1) // 2

    answer = total_possible - m
    print(answer)

main()