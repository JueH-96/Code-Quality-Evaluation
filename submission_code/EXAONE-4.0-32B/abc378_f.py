import collections
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    deg = [0] * (n+1)
    
    index = 1
    for _ in range(n-1):
        u = int(data[index])
        v = int(data[index+1])
        index += 2
        graph[u].append(v)
        graph[v].append(u)
        deg[u] += 1
        deg[v] += 1
        
    dist = [-1] * (n+1)
    from_start = [0] * (n+1)
    q = collections.deque()
    
    for i in range(1, n+1):
        if deg[i] == 2:
            dist[i] = 0
            from_start[i] = i
            q.append(i)
            
    count = 0
    while q:
        u = q.popleft()
        for v in graph[u]:
            if dist[v] == -1:
                if deg[v] == 3:
                    dist[v] = dist[u] + 1
                    from_start[v] = from_start[u]
                    q.append(v)
                elif deg[v] == 2:
                    if from_start[u] < v:
                        count += 1
                        
    print(count)

if __name__ == '__main__':
    main()