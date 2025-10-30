import collections
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    total_nodes = n + m
    graph = [[] for _ in range(total_nodes)]
    
    for i in range(n):
        a = int(next(it))
        set_node = m + i
        for j in range(a):
            x = int(next(it))
            num_node = x - 1
            graph[num_node].append(set_node)
            graph[set_node].append(num_node)
            
    dist = [-1] * total_nodes
    q = collections.deque()
    dist[0] = 0
    q.append(0)
    
    while q:
        u = q.popleft()
        for v in graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
                
    if dist[m-1] == -1:
        print(-1)
    else:
        print(dist[m-1] // 2 - 1)

if __name__ == '__main__':
    main()