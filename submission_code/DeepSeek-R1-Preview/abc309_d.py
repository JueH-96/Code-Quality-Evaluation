from collections import deque
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N1 = int(data[idx])
    idx += 1
    N2 = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    total_nodes = N1 + N2
    adj = [[] for _ in range(total_nodes + 1)]
    
    for _ in range(M):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        adj[a].append(b)
        adj[b].append(a)
    
    # BFS for group 1 (starting from 1)
    distance1 = [-1] * (total_nodes + 1)
    q = deque()
    q.append(1)
    distance1[1] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if distance1[v] == -1:
                distance1[v] = distance1[u] + 1
                q.append(v)
    
    max_d1 = max(distance1[1:N1+1])
    
    # BFS for group 2 (starting from T = N1 + N2)
    T = N1 + N2
    distance2 = [-1] * (total_nodes + 1)
    q = deque()
    q.append(T)
    distance2[T] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if distance2[v] == -1:
                distance2[v] = distance2[u] + 1
                q.append(v)
    
    max_d2 = max(distance2[N1+1 : T+1])
    
    print(max_d1 + max_d2 + 1)

if __name__ == '__main__':
    main()