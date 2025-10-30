import sys
from collections import deque
import bisect

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    K = int(data[index + 2])
    index += 3
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        adj[a].append(b)
        adj[b].append(a)
        index += 2
    
    guards = []
    h_list = []
    for _ in range(K):
        p = int(data[index])
        h = int(data[index + 1])
        guards.append(p)
        h_list.append(h)
        index += 2
    
    INF = N + 1
    distance = [INF] * (N + 1)
    q = deque()
    
    for p in guards:
        distance[p] = 0
        q.append(p)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if distance[v] == INF:
                distance[v] = distance[u] + 1
                q.append(v)
    
    h_list_sorted = sorted(h_list)
    
    guarded = []
    for v in range(1, N + 1):
        if distance[v] < INF:
            idx = bisect.bisect_left(h_list_sorted, distance[v])
            if idx < K:
                guarded.append(v)
    
    guarded.sort()
    print(len(guarded))
    print(' '.join(map(str, guarded)))

if __name__ == '__main__':
    main()