import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    min_xor = float('inf')
    # We'll use BFS to explore all simple paths from 1 to N
    # Each queue element is (current_vertex, visited_mask, current_xor)
    visited = {}
    q = deque()
    start_vertex = 1
    start_mask = 1 << (start_vertex - 1)
    q.append((start_vertex, start_mask, 0))
    
    while q:
        current, mask, xor = q.popleft()
        if current == N:
            if xor < min_xor:
                min_xor = xor
            continue
        for neighbor, w in adj[current]:
            bit = 1 << (neighbor - 1)
            if not (mask & bit):
                new_mask = mask | bit
                new_xor = xor ^ w
                key = (neighbor, new_mask)
                if key not in visited or visited[key] > new_xor:
                    visited[key] = new_xor
                    q.append((neighbor, new_mask, new_xor))
    print(min_xor)

if __name__ == '__main__':
    main()