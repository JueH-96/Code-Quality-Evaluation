import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    comp_count = n

    def find(x):
        root = x
        while root != parent[root]:
            root = parent[root]
        while x != root:
            next_node = parent[x]
            parent[x] = root
            x = next_node
        return root

    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        comp_count -= 1

    ans = m - (n - comp_count)
    print(ans)

if __name__ == "__main__":
    main()