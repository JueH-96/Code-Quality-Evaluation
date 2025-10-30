import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    C = list(map(int, sys.stdin.readline().split()))
    C = [0] + C  # 1-based indexing

    # Compute sum_subtree and parent array
    parent = [0]*(N+1)
    sum_subtree = [0]*(N+1)
    stack = [(1, -1, False)]
    while stack:
        node, p, visited = stack.pop()
        if not visited:
            parent[node] = p
            stack.append((node, p, True))
            for neighbor in adj[node]:
                if neighbor != p:
                    stack.append((neighbor, node, False))
        else:
            sum_subtree[node] = C[node]
            for neighbor in adj[node]:
                if neighbor != parent[node]:
                    sum_subtree[node] += sum_subtree[neighbor]

    # Find centroid
    total_C = sum(C[1:])
    current = 1
    while True:
        found = False
        for neighbor in adj[current]:
            if neighbor == parent[current]:
                sum_v = total_C - sum_subtree[current]
            else:
                sum_v = sum_subtree[neighbor]
            if sum_v > total_C / 2:
                current = neighbor
                found = True
                break
        if not found:
            break

    # Calculate the minimal sum using BFS from centroid
    visited = [False]*(N+1)
    q = deque()
    q.append((current, 0))
    visited[current] = True
    total = 0
    while q:
        node, dist = q.popleft()
        total += C[node] * dist
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append((neighbor, dist + 1))
    print(total)

if __name__ == '__main__':
    main()