import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    adj = [[] for _ in range(N + 1)]
    sum_edges = 0
    for _ in range(N - 1):
        A = int(input[ptr])
        B = int(input[ptr+1])
        C = int(input[ptr+2])
        adj[A].append((B, C))
        adj[B].append((A, C))
        sum_edges += C
        ptr += 3
    
    def get_farthest(start):
        visited = [False] * (N + 1)
        max_dist = 0
        farthest_node = start
        q = deque()
        q.append((start, 0))
        visited[start] = True
        while q:
            current, dist = q.popleft()
            for neighbor, cost in adj[current]:
                if not visited[neighbor]:
                    new_dist = dist + cost
                    if new_dist > max_dist:
                        max_dist = new_dist
                        farthest_node = neighbor
                    visited[neighbor] = True
                    q.append((neighbor, new_dist))
        return farthest_node, max_dist
    
    u, _ = get_farthest(1)
    v, diameter = get_farthest(u)
    print(2 * sum_edges - diameter)

if __name__ == '__main__':
    main()