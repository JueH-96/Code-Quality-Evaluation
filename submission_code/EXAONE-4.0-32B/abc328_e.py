import itertools
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    K_val = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        edges.append((u, v, w))
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    min_cost = None

    for cand in itertools.combinations(edges, n - 1):
        parent = list(range(n + 1))
        rank = [0] * (n + 1)
        total_weight = 0
        valid = True
        for e in cand:
            u, v, w = e
            ru = find(u, parent)
            rv = find(v, parent)
            if ru == rv:
                valid = False
                break
            total_weight += w
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[ru] > rank[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
        
        if valid:
            cost = total_weight % K_val
            if min_cost is None or cost < min_cost:
                min_cost = cost
    
    print(min_cost)

if __name__ == "__main__":
    main()