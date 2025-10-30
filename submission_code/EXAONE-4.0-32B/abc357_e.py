import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    next_node = [x-1 for x in a]
    
    indegree = [0] * n
    for i in range(n):
        j = next_node[i]
        indegree[j] += 1
        
    indegree_work = indegree.copy()
    q = deque()
    for i in range(n):
        if indegree_work[i] == 0:
            q.append(i)
            
    while q:
        u = q.popleft()
        v = next_node[u]
        indegree_work[v] -= 1
        if indegree_work[v] == 0:
            q.append(v)
            
    cycle_size = [0] * n
    visited_cycle = [False] * n
    for i in range(n):
        if indegree_work[i] > 0 and not visited_cycle[i]:
            cycle_nodes = []
            cur = i
            while not visited_cycle[cur]:
                visited_cycle[cur] = True
                cycle_nodes.append(cur)
                cur = next_node[cur]
            sz = len(cycle_nodes)
            for node in cycle_nodes:
                cycle_size[node] = sz
                
    rev_graph = [[] for _ in range(n)]
    for u in range(n):
        v = next_node[u]
        rev_graph[v].append(u)
        
    dist = [-1] * n
    q = deque()
    for i in range(n):
        if indegree_work[i] > 0:
            dist[i] = 0
            q.append(i)
            
    while q:
        u = q.popleft()
        for v in rev_graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                cycle_size[v] = cycle_size[u]
                q.append(v)
                
    total = 0
    for i in range(n):
        total += dist[i] + cycle_size[i]
        
    print(total)

if __name__ == '__main__':
    main()