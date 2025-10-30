import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    D = int(data[ptr])
    ptr += 1
    
    X = []
    Y = []
    for _ in range(N):
        x = int(data[ptr])
        ptr += 1
        y = int(data[ptr])
        ptr += 1
        X.append(x)
        Y.append(y)
    
    # Build adjacency list
    adjacency = [[] for _ in range(N)]
    D_squared = D * D
    for i in range(N):
        for j in range(i + 1, N):
            dx = X[i] - X[j]
            dy = Y[i] - Y[j]
            if dx * dx + dy * dy <= D_squared:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # BFS
    infected = [False] * N
    queue = deque()
    queue.append(0)
    infected[0] = True
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if not infected[v]:
                infected[v] = True
                queue.append(v)
    
    # Output
    for i in range(N):
        print("Yes" if infected[i] else "No")

if __name__ == '__main__':
    main()