# YOUR CODE HERE
import sys
import sys
import sys
from collections import deque

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N1 = int(data[idx]); idx +=1
    N2 = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    N = N1 + N2
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(data[idx]); idx +=1
        b = int(data[idx]); idx +=1
        adj[a].append(b)
        adj[b].append(a)
    
    def bfs(start):
        distance = [0]*(N+1)
        q = deque()
        q.append(start)
        distance[start] = 1
        while q:
            u = q.popleft()
            for v in adj[u]:
                if distance[v] == 0:
                    distance[v] = distance[u] +1
                    q.append(v)
        return distance
    
    distance1 = bfs(1)
    distance2 = bfs(N)
    
    max_dist1 = 0
    for u in range(1, N1+1):
        if distance1[u] > max_dist1:
            max_dist1 = distance1[u]
    max_dist2 = 0
    for v in range(N1+1, N+1):
        if distance2[v] > max_dist2:
            max_dist2 = distance2[v]
    print(max_dist1 + max_dist2)
    
if __name__ == "__main__":
    main()