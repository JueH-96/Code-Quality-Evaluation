import sys
from collections import defaultdict, deque

def main() -> None:
    N = int(sys.stdin.readline())
    
    # build graph
    out_deg = defaultdict(int)
    in_deg  = defaultdict(int)
    
    edges_by_comp = defaultdict(list)      # for weak components later
    parent = {c: c for c in range(26)}     # Union–Find over 26 letters
    
    def idx(ch):            # map 'A'..'Z' → 0..25
        return ord(ch) - 65
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def unite(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    
    names = [sys.stdin.readline().strip() for _ in range(N)]
    
    for s in names:
        a, b = idx(s[0]), idx(s[1])
        out_deg[a] += 1
        in_deg[b]  += 1
        unite(a, b)                    # undirected union
    
    # collect vertices actually appearing
    present = set()
    for v in range(26):
        if out_deg[v] or in_deg[v]:
            present.add(v)
    
    # group vertices by weak component
    comps = defaultdict(list)
    for v in present:
        comps[find(v)].append(v)
    
    answer = 0
    for comp_vertices in comps.values():
        surplus = 0
        for v in comp_vertices:
            surplus += max(0, out_deg[v] - in_deg[v])
        answer += max(surplus, 1)      # rule derived above
    
    print(answer)

if __name__ == "__main__":
    main()