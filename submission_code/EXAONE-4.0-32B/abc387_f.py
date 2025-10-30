import sys
from collections import deque

mod = 998244353

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    A = [a-1 for a in A]

    visited = [False] * n
    cycle_rep = [-1] * n
    for i in range(n):
        if not visited[i]:
            path = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                path.append(cur)
                cur = A[cur]
            if cur in path:
                idx = path.index(cur)
                cycle_nodes = path[idx:]
                rep_node = min(cycle_nodes)
                for node in cycle_nodes:
                    cycle_rep[node] = rep_node

    rev = [[] for _ in range(n)]
    for i in range(n):
        j = A[i]
        rev[j].append(i)
    
    rep_arr = [-1] * n
    q = deque()
    for i in range(n):
        if cycle_rep[i] != -1:
            rep_arr[i] = cycle_rep[i]
            q.append(i)
            
    while q:
        u = q.popleft()
        for v in rev[u]:
            if rep_arr[v] == -1:
                rep_arr[v] = rep_arr[u]
                q.append(v)
                
    in_tree = [False] * n
    for i in range(n):
        if cycle_rep[i] != -1:
            if i == cycle_rep[i]:
                in_tree[i] = True
            else:
                in_tree[i] = False
        else:
            in_tree[i] = True
            
    children = [[] for _ in range(n)]
    for i in range(n):
        if in_tree[i]:
            j = A[i]
            if in_tree[j]:
                parent_candidate = j
            else:
                parent_candidate = rep_arr[j]
            if parent_candidate != i:
                children[parent_candidate].append(i)
                
    indegree = [0] * n
    for i in range(n):
        if in_tree[i]:
            for child in children[i]:
                indegree[child] += 1
                
    roots = [i for i in range(n) if in_tree[i] and indegree[i] == 0]
    
    q = deque(roots)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)
                
    order.reverse()
    
    dp = [[0] * (m+1) for _ in range(n)]
    
    for u in order:
        if not children[u]:
            for k in range(1, m+1):
                dp[u][k] = 1
        else:
            prod_arr = [1] * (m+1)
            for v in children[u]:
                prefix = [0] * (m+1)
                s = 0
                for k in range(1, m+1):
                    s = (s + dp[v][k]) % mod
                    prefix[k] = s
                for k in range(1, m+1):
                    prod_arr[k] = (prod_arr[k] * prefix[k]) % mod
            for k in range(1, m+1):
                dp[u][k] = prod_arr[k]
                
    ans = 0
    for u in roots:
        for k in range(1, m+1):
            ans = (ans + dp[u][k]) % mod
            
    print(ans)

if __name__ == '__main__':
    main()