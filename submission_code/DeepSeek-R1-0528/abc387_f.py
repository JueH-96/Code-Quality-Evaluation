mod = 998244353

import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A_input = list(map(int, data[2:2+n]))
    A_arr = [0] * (n+1)
    for i in range(1, n+1):
        A_arr[i] = A_input[i-1]
    
    in_deg = [0] * (n+1)
    for i in range(1, n+1):
        j = A_arr[i]
        in_deg[j] += 1
        
    non_cycle = set()
    cycle = set()
    q = deque()
    for i in range(1, n+1):
        if in_deg[i] == 0:
            q.append(i)
            
    while q:
        u = q.popleft()
        non_cycle.add(u)
        v = A_arr[u]
        in_deg[v] -= 1
        if in_deg[v] == 0:
            q.append(v)
            
    for i in range(1, n+1):
        if i not in non_cycle:
            cycle.add(i)
            
    rev_graph = [[] for _ in range(n+1)]
    for i in range(1, n+1):
        j = A_arr[i]
        rev_graph[j].append(i)
        
    tree_children = [[] for _ in range(n+1)]
    for i in range(1, n+1):
        for child in rev_graph[i]:
            if child in non_cycle:
                tree_children[i].append(child)
                
    child_waiting = [0] * (n+1)
    prefix_arr = [[0]*(m+1) for _ in range(n+1)]
    
    for u in range(1, n+1):
        if u in non_cycle:
            child_waiting[u] = len(tree_children[u])
            
    q_nc = deque()
    for u in non_cycle:
        if child_waiting[u] == 0:
            q_nc.append(u)
            
    while q_nc:
        u = q_nc.popleft()
        f = [0] * (m+1)
        for a in range(1, m+1):
            f[a] = 1
            
        for v in tree_children[u]:
            for a in range(1, m+1):
                f[a] = (f[a] * prefix_arr[v][a]) % mod
                
        for a in range(1, m+1):
            prefix_arr[u][a] = (prefix_arr[u][a-1] + f[a]) % mod
            
        p = A_arr[u]
        if p in non_cycle:
            child_waiting[p] -= 1
            if child_waiting[p] == 0:
                q_nc.append(p)
                
    dp_cycle_arr = [[0]*(m+1) for _ in range(n+1)]
    for u in cycle:
        g = [0] * (m+1)
        for a in range(1, m+1):
            g[a] = 1
        for v in tree_children[u]:
            for a in range(1, m+1):
                g[a] = (g[a] * prefix_arr[v][a]) % mod
        dp_cycle_arr[u] = g
        
    visited_cycle = [False] * (n+1)
    cycles_list = []
    for u in cycle:
        if not visited_cycle[u]:
            comp = []
            cur = u
            while not visited_cycle[cur]:
                visited_cycle[cur] = True
                comp.append(cur)
                cur = A_arr[cur]
            cycles_list.append(comp)
            
    ans = 1
    for comp in cycles_list:
        total_comp = 0
        for a0 in range(1, m+1):
            prod = 1
            for u in comp:
                prod = (prod * dp_cycle_arr[u][a0]) % mod
            total_comp = (total_comp + prod) % mod
        ans = (ans * total_comp) % mod
        
    print(ans)
    
if __name__ == "__main__":
    main()