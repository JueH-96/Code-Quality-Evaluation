import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    
    parent = list(range(n+1))
    rank = [0] * (n+1)
    top = [None] * (n+1)
    for i in range(1, n+1):
        top[i] = [i]
    
    def find(x):
        root = x
        while root != parent[root]:
            root = parent[root]
        current = x
        while current != root:
            nxt = parent[current]
            parent[current] = root
            current = nxt
        return root
    
    out_lines = []
    for _ in range(q):
        typ = next(it)
        if typ == '1':
            u = int(next(it))
            v = int(next(it))
            ru = find(u)
            rv = find(v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
            combined = top[ru] + top[rv]
            combined.sort(reverse=True)
            top[ru] = combined[:10]
        else:
            v = int(next(it))
            k = int(next(it))
            r = find(v)
            if len(top[r]) < k:
                out_lines.append("-1")
            else:
                out_lines.append(str(top[r][k-1]))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()