import sys
sys.setrecursionlimit(10**6)

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); q = int(next(it))
    journeys = []
    for _ in range(m):
        s = int(next(it)); t = int(next(it))
        journeys.append((s, t))
    
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
            
    comp_journeys = [[] for _ in range(n+1)]
    for idx, (s, t) in enumerate(journeys, 1):
        l = min(s, t)
        r = max(s, t)
        if s < t:
            typ = 0
        else:
            typ = 1
        union(l, r)
        root = find(l)
        comp_journeys[root].append((idx, l, r, typ))
        
    comp_witness_pairs = []
    for root in range(1, n+1):
        comp_list = comp_journeys[root]
        if not comp_list:
            continue
        type0 = []
        type1 = []
        for (idx, l, r, typ) in comp_list:
            if typ == 0:
                if r - l > 1:
                    type0.append((idx, l+1, r-1))
            else:
                if r - l > 1:
                    type1.append((idx, l+1, r-1))
                    
        if not type0 or not type1:
            continue
            
        events = []
        for idx, a, b in type0:
            events.append((a, 0, idx))
            events.append((b+1, 1, idx))
        for idx, a, b in type1:
            events.append((a, 0, idx))
            events.append((b+1, 1, idx))
            
        events.sort(key=lambda x: (x[0], x[1]))
        
        active0 = set()
        active1 = set()
        found_in_comp = False
        for event in events:
            pos, etype, idx_val = event
            if etype == 0:
                if idx_val in [item[0] for item in type0]:
                    active0.add(idx_val)
                else:
                    active1.add(idx_val)
            else:
                if idx_val in [item[0] for item in type0]:
                    if idx_val in active0:
                        active0.remove(idx_val)
                else:
                    if idx_val in active1:
                        active1.remove(idx_val)
            if active0 and active1:
                i0 = min(active0)
                j0 = min(active1)
                a_val = min(i0, j0)
                b_val = max(i0, j0)
                comp_witness_pairs.append((a_val, b_val))
                found_in_comp = True
                break
                
    F = [0] * (m+1)
    for a, b in comp_witness_pairs:
        if a <= b:
            if a > F[b]:
                F[b] = a
                
    for i in range(1, m+1):
        if F[i] < F[i-1]:
            F[i] = F[i-1]
            
    out_lines = []
    for _ in range(q):
        L = int(next(it)); R = int(next(it))
        if F[R] >= L:
            out_lines.append("No")
        else:
            out_lines.append("Yes")
            
    print("
".join(out_lines))

if __name__ == "__main__":
    main()