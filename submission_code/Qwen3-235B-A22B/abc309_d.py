import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N1 = int(input[ptr])
    ptr += 1
    N2 = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    S = N1 + N2
    adj = [[] for _ in range(S + 1)]  # 1-based indexing

    for _ in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        adj[a].append(b)
        adj[b].append(a)

    # BFS for first component (node 1)
    dist1 = [-1] * (S + 1)
    dist1[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                q.append(v)

    max_d1 = 0
    for i in range(1, N1 + 1):
        if dist1[i] > max_d1:
            max_d1 = dist1[i]

    # BFS for second component (node S)
    dist2 = [-1] * (S + 1)
    dist2[S] = 0
    q = deque([S])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                q.append(v)

    max_d2 = 0
    for i in range(N1 + 1, S + 1):
        if dist2[i] > max_d2:
            max_d2 = dist2[i]

    print(max_d1 + max_d2 + 1)

if __name__ == "__main__":
    main()