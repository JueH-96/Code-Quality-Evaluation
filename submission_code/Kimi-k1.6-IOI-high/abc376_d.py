from collections import deque
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    M = int(input[idx])
    idx +=1
    
    adj = [[] for _ in range(N+1)]
    edges = []
    for _ in range(M):
        a = int(input[idx])
        idx +=1
        b = int(input[idx])
        idx +=1
        adj[a].append(b)
        edges.append((a, b))
    
    distance = [-1] * (N +1)
    distance[1] =0
    q = deque()
    q.append(1)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if distance[v] == -1:
                distance[v] = distance[u] +1
                q.append(v)
    
    candidates = []
    for a, b in edges:
        if b ==1 and distance[a] != -1:
            candidates.append(distance[a] +1)
    
    if candidates:
        print(min(candidates))
    else:
        print(-1)

if __name__ == "__main__":
    main()