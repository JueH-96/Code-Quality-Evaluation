import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    graph = [[] for _ in range(n)]
    index = 1
    for i in range(n-1):
        a, b = map(int, data[index].split())
        index += 1
        a -= 1
        b -= 1
        graph[a].append(b)
        graph[b].append(a)
    
    C = list(map(int, data[index].split()))
    total_weight = sum(C)
    
    parent = [-1] * n
    dist0 = [-1] * n
    children = [[] for _ in range(n)]
    order = []
    q = deque([0])
    dist0[0] = 0
    parent[0] = -1
    
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            dist0[v] = dist0[u] + 1
            children[u].append(v)
            q.append(v)
            
    subtree_sum = [0] * n
    for i in range(n):
        subtree_sum[i] = C[i]
        
    for i in range(len(order)-1, -1, -1):
        u = order[i]
        for v in children[u]:
            subtree_sum[u] += subtree_sum[v]
            
    F0 = 0
    for i in range(n):
        F0 += C[i] * dist0[i]
        
    F_vals = [0] * n
    F_vals[0] = F0
    ans = F0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in children[u]:
            F_vals[v] = F_vals[u] + total_weight - 2 * subtree_sum[v]
            if F_vals[v] < ans:
                ans = F_vals[v]
            q.append(v)
            
    print(ans)

if __name__ == "__main__":
    main()