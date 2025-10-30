import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    K = int(data[idx+2])
    idx += 3
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a = int(data[idx])
        b = int(data[idx+1])
        adj[a].append(b)
        adj[b].append(a)
        idx += 2
    guards = []
    for _ in range(K):
        p = int(data[idx])
        h = int(data[idx+1])
        guards.append((p, h))
        idx += 2
    guarded = set()
    for p, h in guards:
        if p in guarded:
            continue
        q = deque()
        q.append((p, 0))
        visited = {p: 0}
        while q:
            node, dist = q.popleft()
            if dist >= h:
                continue
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    q.append((neighbor, dist + 1))
        for node in visited:
            guarded.add(node)
    guarded = sorted(guarded)
    print(len(guarded))
    print(' '.join(map(str, guarded)))

if __name__ == "__main__":
    main()