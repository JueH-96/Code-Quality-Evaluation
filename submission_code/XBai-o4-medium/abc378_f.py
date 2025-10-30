import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    adj = [set() for _ in range(N + 1)]
    for _ in range(N - 1):
        u = int(input[ptr])
        v = int(input[ptr + 1])
        ptr += 2
        adj[u].add(v)
        adj[v].add(u)
    
    degree = [0] * (N + 1)
    for i in range(1, N + 1):
        degree[i] = len(adj[i])
    
    ans = 0
    
    for u in range(1, N + 1):
        if degree[u] != 3:
            for v in adj[u]:
                if degree[v] != 3:
                    continue
                else:
                    queue = deque()
                    queue.append((v, u))
                    end_points = []
                    while queue:
                        current, prev = queue.popleft()
                        if degree[current] != 3:
                            end_points.append(current)
                        else:
                            for neighbor in adj[current]:
                                if neighbor != prev:
                                    queue.append((neighbor, current))
                    for y in end_points:
                        if degree[u] == 2 and degree[y] == 2 and u < y:
                            ans += 1
    print(ans)

if __name__ == '__main__':
    main()