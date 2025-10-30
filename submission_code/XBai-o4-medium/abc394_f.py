import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    adj = [[] for _ in range(N + 1)]
    degrees = [0] * (N + 1)
    
    for _ in range(N - 1):
        a = int(input[ptr])
        b = int(input[ptr + 1])
        ptr += 2
        adj[a].append(b)
        adj[b].append(a)
        degrees[a] += 1
        degrees[b] += 1
    
    valid = [i for i in range(1, N + 1) if degrees[i] >= 4]
    if not valid:
        print(-1)
        return
    
    visited = [False] * (N + 1)
    max_comp = 0
    
    for u in range(1, N + 1):
        if degrees[u] >= 4 and not visited[u]:
            q = deque()
            q.append(u)
            visited[u] = True
            current_count = 1
            while q:
                current = q.popleft()
                for neighbor in adj[current]:
                    if not visited[neighbor] and degrees[neighbor] >= 4:
                        visited[neighbor] = True
                        q.append(neighbor)
                        current_count += 1
            if current_count > max_comp:
                max_comp = current_count
    
    print(3 * max_comp + 2)

if __name__ == "__main__":
    main()