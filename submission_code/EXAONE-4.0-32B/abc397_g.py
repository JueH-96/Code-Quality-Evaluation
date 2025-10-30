import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    k = int(data[2])
    edges = []
    index = 3
    for i in range(m):
        u = int(data[index])
        v = int(data[index+1])
        index += 2
        edges.append((u, v))
    
    out_deg = [0] * (n+1)
    in_edges = [[] for _ in range(n+1)]
    for idx, (u, v) in enumerate(edges):
        out_deg[u] += 1
        in_edges[v].append((u, idx))
        
    def check(x):
        if x == 0:
            return True
            
        unavoidable = [[False] * (x+1) for _ in range(n+1)]
        ZUV = [[[False] * m for _ in range(x+1)] for __ in range(n+1)]
        ONUV = [[[False] * m for _ in range(x+1)] for __ in range(n+1)]
        constraint_count = [[0] * (x+1) for _ in range(n+1)]
        
        q = deque()
        for h in range(x):
            unavoidable[n][h] = True
            q.append((n, h))
            
        while q:
            u, h_val = q.popleft()
            for (v, idx) in in_edges[u]:
                if h_val <= x:
                    if not ZUV[v][h_val][idx]:
                        ZUV[v][h_val][idx] = True
                        if ONUV[v][h_val][idx]:
                            constraint_count[v][h_val] += 1
                            if constraint_count[v][h_val] == out_deg[v] and not unavoidable[v][h_val]:
                                unavoidable[v][h_val] = True
                                q.append((v, h_val))
                if h_val >= 1:
                    if not ONUV[v][h_val-1][idx]:
                        ONUV[v][h_val-1][idx] = True
                        if ZUV[v][h_val-1][idx]:
                            constraint_count[v][h_val-1] += 1
                            if constraint_count[v][h_val-1] == out_deg[v] and not unavoidable[v][h_val-1]:
                                unavoidable[v][h_val-1] = True
                                q.append((v, h_val-1))
                                
        if unavoidable[1][0]:
            return False
            
        must_ones = 0
        must_zeros = 0
        conflict = False
        for idx in range(m):
            u, v_target = edges[idx]
            found_zero_constraint = False
            found_one_constraint = False
            for h_val in range(x+1):
                if unavoidable[u][h_val]:
                    continue
                if ZUV[u][h_val][idx]:
                    found_zero_constraint = True
                if ONUV[u][h_val][idx]:
                    found_one_constraint = True
            if found_zero_constraint and found_one_constraint:
                conflict = True
                break
            elif found_zero_constraint:
                must_ones += 1
            elif found_one_constraint:
                must_zeros += 1
                
        if conflict:
            return False
            
        optional = m - must_ones - must_zeros
        if must_ones <= k <= must_ones + optional:
            return True
        else:
            return False
            
    for x in range(k, -1, -1):
        if check(x):
            print(x)
            return
    print(0)

if __name__ == "__main__":
    main()