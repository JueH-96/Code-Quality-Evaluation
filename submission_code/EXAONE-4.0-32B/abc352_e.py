import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    ops = []
    for _ in range(m):
        k = int(next(it))
        c = int(next(it))
        arr = [int(next(it)) for _ in range(k)]
        ops.append((c, k, arr))
    
    ops.sort(key=lambda x: x[0])
    
    parent = list(range(n+1))
    rank = [0] * (n+1)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        x = find(x)
        y = find(y)
        if x == y:
            return
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1
            
    total_cost = 0
    for c, k, arr in ops:
        comp_set = set()
        for node in arr:
            comp_set.add(find(node))
        if len(comp_set) <= 1:
            continue
        total_cost += c * (len(comp_set) - 1)
        base = comp_set.pop()
        for root in comp_set:
            union(base, root)
            
    roots = set()
    for i in range(1, n+1):
        roots.add(find(i))
    print(-1 if len(roots) != 1 else total_cost)

if __name__ == "__main__":
    main()