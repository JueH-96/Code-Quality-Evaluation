import itertools
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    k_val = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        edges.append((u, v, w))
    
    if m == n - 1:
        total = sum(w for _, _, w in edges)
        print(total % k_val)
        return
        
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    min_cost = k_val
    for comb in itertools.combinations(range(m), n - 1):
        for i in range(1, n + 1):
            parent[i] = i
            rank[i] = 0
            
        total_weight = 0
        is_tree = True
        for idx in comb:
            u, v, w_val = edges[idx]
            total_weight += w_val
            ru = find(u)
            rv = find(v)
            if ru == rv:
                is_tree = False
                break
            else:
                if rank[ru] < rank[rv]:
                    parent[ru] = rv
                elif rank[ru] > rank[rv]:
                    parent[rv] = ru
                else:
                    parent[rv] = ru
                    rank[ru] += 1
                    
        if is_tree:
            residue = total_weight % k_val
            if residue < min_cost:
                min_cost = residue
                if min_cost == 0:
                    break
                    
    print(min_cost)

if __name__ == '__main__':
    main()