import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    def find(x):
        root = x
        while root != parent[root]:
            root = parent[root]
        while x != root:
            next_node = parent[x]
            parent[x] = root
            x = next_node
        return root

    def union(a, b):
        a = find(a)
        b = find(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            parent[a] = b
        elif rank[a] > rank[b]:
            parent[b] = a
        else:
            parent[b] = a
            rank[a] += 1

    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        if u == v:
            continue
        ru = find(u)
        rv = find(v)
        if ru != rv:
            union(ru, rv)
    
    k = int(next(it))
    forbidden_set = set()
    for _ in range(k):
        x = int(next(it))
        y = int(next(it))
        rx = find(x)
        ry = find(y)
        if rx == ry:
            continue
        if rx > ry:
            rx, ry = ry, rx
        forbidden_set.add((rx, ry))
    
    q = int(next(it))
    output_lines = []
    for _ in range(q):
        p = int(next(it))
        q_val = int(next(it))
        rp = find(p)
        rq = find(q_val)
        if rp == rq:
            output_lines.append("Yes")
        else:
            if rp > rq:
                rp, rq = rq, rp
            output_lines.append("No" if (rp, rq) in forbidden_set else "Yes")
    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()