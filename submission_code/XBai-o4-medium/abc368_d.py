import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, K = int(input[ptr]), int(input[ptr+1])
    ptr += 2
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a = int(input[ptr])
        b = int(input[ptr + 1])
        adj[a].append(b)
        adj[b].append(a)
        ptr += 2
    
    required = [False] * (N + 1)
    V = list(map(int, input[ptr:ptr + K]))
    for v in V:
        required[v] = True
    
    degree = [0] * (N + 1)
    for i in range(1, N + 1):
        degree[i] = len(adj[i])
    
    q = deque()
    for i in range(1, N + 1):
        if degree[i] == 1 and not required[i]:
            q.append(i)
    
    removed = [False] * (N + 1)
    removed_count = 0
    
    while q:
        u = q.popleft()
        if removed[u]:
            continue
        removed[u] = True
        removed_count += 1
        for v in adj[u]:
            if not removed[v]:
                degree[v] -= 1
                if degree[v] == 1 and not required[v]:
                    q.append(v)
    
    print(N - removed_count)

if __name__ == "__main__":
    main()