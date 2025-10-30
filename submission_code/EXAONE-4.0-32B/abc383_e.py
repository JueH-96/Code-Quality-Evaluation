import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        edges.append((w, u, v))
    
    A_list = [int(next(it)) - 1 for _ in range(k)]
    B_list = [int(next(it)) - 1 for _ in range(k)]
    
    cntA = [0] * n
    for a_val in A_list:
        cntA[a_val] += 1
    cntB = [0] * n
    for b_val in B_list:
        cntB[b_val] += 1
        
    parent = list(range(n))
    size = [1] * n
    a_count = cntA[:]
    b_count = cntB[:]
    
    def find(x):
        stack = []
        while parent[x] != x:
            stack.append(x)
            x = parent[x]
        for i in stack:
            parent[i] = x
        return x
            
    edges.sort()
    total_cost = 0
    
    for w, u, v in edges:
        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue
            
        if size[ru] < size[rv]:
            ru, rv = rv, ru
            
        match_ru = min(a_count[ru], b_count[ru])
        a_ru = a_count[ru] - match_ru
        b_ru = b_count[ru] - match_ru
        
        match_rv = min(a_count[rv], b_count[rv])
        a_rv = a_count[rv] - match_rv
        b_rv = b_count[rv] - match_rv
        
        part1 = min(a_ru, b_rv)
        part2 = min(b_ru, a_rv)
        across = part1 + part2
        total_cost += w * across
        
        a_new = a_ru + a_rv - part1 - part2
        b_new = b_ru + b_rv - part1 - part2
        
        parent[rv] = ru
        size[ru] += size[rv]
        a_count[ru] = a_new
        b_count[ru] = b_new
        
    print(total_cost)

if __name__ == "__main__":
    main()