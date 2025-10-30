import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx +=1
    
    adj = [[] for _ in range(N+1)]
    sum_edges =0
    for _ in range(N-1):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        idx +=3
        adj[A].append( (B, C) )
        adj[B].append( (A, C) )
        sum_edges += C
    
    def bfs(start):
        queue = deque()
        queue.append(start)
        distance = [-1]*(N+1)
        distance[start] =0
        max_dist =0
        max_node = start
        while queue:
            u = queue.popleft()
            for v, w in adj[u]:
                if distance[v] == -1:
                    distance[v] = distance[u] +w
                    if distance[v] > max_dist:
                        max_dist = distance[v]
                        max_node =v
                    queue.append(v)
        return max_node, max_dist
    
    a, d1 = bfs(1)
    b, diameter = bfs(a)
    
    print(2 * sum_edges - diameter)

if __name__ == '__main__':
    main()