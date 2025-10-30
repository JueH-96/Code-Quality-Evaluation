import sys
import threading

def main():
    import sys

    data = sys.stdin
    line = data.readline().split()
    if not line:
        # No input
        print(0)
        return
    n = int(line[0])
    m = int(line[1])

    parent = list(range(n+1))
    # Optional: union by size
    size = [1] * (n+1)

    def find(x):
        # iterative path-compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = 0
    for _ in range(m):
        u, v = map(int, data.readline().split())
        ru = find(u)
        rv = find(v)
        if ru != rv:
            # union by size
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]
            merges += 1
        # else: this edge would form a cycle

    # The minimum number of edges to delete to make the graph acyclic
    # is exactly the number of edges that would form cycles,
    # i.e. total edges minus number of successful merges.
    print(m - merges)

if __name__ == "__main__":
    main()