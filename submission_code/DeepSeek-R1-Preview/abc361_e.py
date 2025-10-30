import sys
from collections import deque

def find_farthest(start, adj):
    max_dist = 0
    far_node = start
    parent = {}
    queue = deque()
    queue.append((start, 0))
    parent[start] = -1
    while queue:
        node, dist = queue.popleft()
        if dist > max_dist:
            max_dist = dist
            far_node = node
        for neighbor, weight in adj[node]:
            if parent[node] != neighbor:
                parent[neighbor] = node
                queue.append((neighbor, dist + weight))
    return far_node, max_dist

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    adj = [[] for _ in range(N + 1)]
    sum_C = 0
    for _ in range(N - 1):
        A = int(input[idx])
        B = int(input[idx+1])
        C = int(input[idx+2])
        adj[A].append((B, C))
        adj[B].append((A, C))
        sum_C += C
        idx += 3
    if N == 1:
        print(0)
        return
    # Find the diameter
    v, _ = find_farthest(1, adj)
    w, diameter = find_farthest(v, adj)
    ans = 2 * sum_C - diameter
    print(ans)

if __name__ == '__main__':
    main()