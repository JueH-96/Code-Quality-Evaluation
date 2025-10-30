import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    degrees = [0] * (N + 1)
    for i in range(1, N+1):
        degrees[i] = len(edges[i])
    
    S = [i for i in range(1, N+1) if degrees[i] == 2]
    count = 0

    for i in range(len(S)):
        u = S[i]
        for j in range(i+1, len(S)):
            v = S[j]
            visited = set()
            queue = deque()
            queue.append((u, [u]))
            found = False
            while queue:
                node, path = queue.popleft()
                if node == v:
                    valid = True
                    for n in path[1:-1]:
                        if degrees[n] != 3:
                            valid = False
                            break
                    if valid:
                        count += 1
                    found = True
                    break
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in edges[node]:
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
            if found:
                continue

    print(count)

if __name__ == '__main__':
    main()