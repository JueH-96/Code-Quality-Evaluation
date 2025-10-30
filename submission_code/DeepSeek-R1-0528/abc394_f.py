import heapq
from collections import deque
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    graph = [[] for _ in range(n)]
    index = 1
    for i in range(n-1):
        a = int(data[index]); b = int(data[index+1]); index += 2
        a -= 1
        b -= 1
        graph[a].append(b)
        graph[b].append(a)
    
    parent = [-1] * n
    children = [[] for _ in range(n)]
    q = deque([0])
    parent[0] = -1
    visited = [False] * n
    visited[0] = True
    while q:
        u = q.popleft()
        for v in graph[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                children[u].append(v)
                q.append(v)
                
    down_f = [0] * n
    chain = [1] * n
    
    order = []
    stack = [0]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            stack.append(v)
    order.reverse()
    
    for u in order:
        if len(children[u]) < 3:
            down_f[u] = 0
            chain[u] = 1
        else:
            child_chains = []
            for v in children[u]:
                if down_f[v] > 0:
                    child_chains.append(down_f[v])
                else:
                    child_chains.append(1)
            top3 = heapq.nlargest(3, child_chains)
            down_f[u] = 1 + top3[0] + top3[1] + top3[2]
            chain[u] = down_f[u]
            
    up_chain = [0] * n
    up_chain[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        branches = []
        if u != 0:
            branches.append(up_chain[u])
        for v in children[u]:
            branches.append(chain[v])
        deg_u = len(branches)
        if deg_u - 1 < 3:
            for v in children[u]:
                up_chain[v] = 1
                q.append(v)
        else:
            top4 = heapq.nlargest(4, branches)
            total_top3 = top4[0] + top4[1] + top4[2]
            fourth = top4[3]
            for v in children[u]:
                x = chain[v]
                if x <= fourth:
                    total_after = total_top3
                else:
                    total_after = total_top3 - x + fourth
                g_u = 1 + total_after
                up_chain[v] = g_u
                q.append(v)
                
    ans = -1
    for u in range(n):
        deg = len(graph[u])
        if deg < 4:
            continue
        branches = []
        if u != 0:
            branches.append(up_chain[u])
        for v in children[u]:
            branches.append(chain[v])
        if len(branches) < 4:
            continue
        top4 = heapq.nlargest(4, branches)
        total = top4[0] + top4[1] + top4[2] + top4[3]
        candidate = 1 + total
        if candidate > ans:
            ans = candidate
            
    print(ans if ans != -1 else -1)

if __name__ == '__main__':
    main()