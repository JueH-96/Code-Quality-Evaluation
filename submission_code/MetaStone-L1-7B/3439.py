import sys
from collections import deque

def compute_diameter(tree, n):
    def bfs(start):
        visited = {}
        q = deque()
        q.append((start, 0))
        visited[start] = 0
        max_dist = 0
        far_node = start
        while q:
            current, dist = q.popleft()
            for neighbor in tree[current]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    if visited[neighbor] > max_dist:
                        max_dist = visited[neighbor]
                        far_node = neighbor
                    q.append((neighbor, dist + 1))
        return far_node, max_dist

    if n == 1:
        return 0
    u, _ = bfs(0)
    v, d = bfs(u)
    return d

def main():
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    m1 = int(input[ptr])
    ptr += 1
    edges1 = [[] for _ in range(n)]
    for _ in range(m1):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        edges1[a].append(b)
        edges1[b].append(a)
    d1 = compute_diameter(edges1, n)
    m2 = int(input[ptr])
    ptr += 1
    edges2 = [[] for _ in range(n)]
    for _ in range(m2):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        edges2[a].append(b)
        edges2[b].append(a)
    d2 = compute_diameter(edges2, n)
    r1 = (d1 + 1) // 2
    r2 = (d2 + 1) // 2
    new_d = max(d1, d2, r1 + r2 + 1)
    print(new_d)

if __name__ == "__main__":
    main()