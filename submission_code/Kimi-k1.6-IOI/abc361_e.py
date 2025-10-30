import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    adj = [[] for _ in range(N + 1)]
    sum_C = 0
    for _ in range(N - 1):
        A = int(input[idx])
        B = int(input[idx + 1])
        C = int(input[idx + 2])
        adj[A].append((B, C))
        adj[B].append((A, C))
        sum_C += C
        idx += 3
    
    def find_farthest(start):
        max_dist = 0
        farthest_node = start
        visited = [False] * (N + 1)
        q = deque()
        q.append((start, 0))
        visited[start] = True
        while q:
            node, dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                farthest_node = node
            for neighbor, weight in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append((neighbor, dist + weight))
        return farthest_node, max_dist
    
    u, _ = find_farthest(1)
    _, diameter = find_farthest(u)
    print(sum_C * 2 - diameter)

if __name__ == '__main__':
    main()