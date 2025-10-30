import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    parent = list(range(n+1))
    rank = [0] * (n+1)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
            
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        if u == v:
            continue
        union(u, v)
        
    roots = [0] * (n+1)
    for i in range(1, n+1):
        roots[i] = find(i)
        
    k = int(next(it))
    forbidden_set = set()
    for _ in range(k):
        x = int(next(it))
        y = int(next(it))
        rx = roots[x]
        ry = roots[y]
        if rx != ry:
            key = (min(rx, ry), max(rx, ry))
            forbidden_set.add(key)
            
    q = int(next(it))
    output_lines = []
    for _ in range(q):
        p = int(next(it))
        q_val = int(next(it))
        rp = roots[p]
        rq = roots[q_val]
        if rp == rq:
            output_lines.append("Yes")
        else:
            key = (min(rp, rq), max(rp, rq))
            if key in forbidden_set:
                output_lines.append("No")
            else:
                output_lines.append("Yes")
                
    print("
".join(output_lines))

if __name__ == "__main__":
    main()