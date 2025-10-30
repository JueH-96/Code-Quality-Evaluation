import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N1, N2, M = map(int, input().split())
    N = N1 + N2

    # Build adjacency list for the whole graph
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)

    # BFS from node 1 over the left component (1..N1)
    dist1 = [-1] * (N+1)
    dist1[1] = 0
    dq = deque([1])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                dq.append(v)

    # BFS from node N (which is N1+N2) over the right component
    dist2 = [-1] * (N+1)
    end = N
    dist2[end] = 0
    dq = deque([end])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                dq.append(v)

    # Find the farthest distance in the left component from 1
    maxL = 0
    for i in range(1, N1+1):
        if dist1[i] > maxL:
            maxL = dist1[i]

    # Find the farthest distance in the right component from N
    maxR = 0
    for i in range(N1+1, N+1):
        if dist2[i] > maxR:
            maxR = dist2[i]

    # The best we can do is connect the farthest on the left to the farthest on the right
    print(maxL + 1 + maxR)

if __name__ == '__main__':
    main()