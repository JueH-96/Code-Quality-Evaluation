import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    parent = list(range(N))
    size = [1] * N

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
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    # Read edges and union the endpoints
    for _ in range(M):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        union(a, b)

    # Count the size of each component via its root
    comp_count = {}
    for i in range(N):
        r = find(i)
        comp_count[r] = comp_count.get(r, 0) + 1

    # For each component of size s, it can become a clique of s*(s-1)/2 edges.
    # Subtract the M existing edges to get the number of new operations.
    total_possible = 0
    for s in comp_count.values():
        total_possible += s * (s - 1) // 2

    answer = total_possible - M
    print(answer)

if __name__ == "__main__":
    main()