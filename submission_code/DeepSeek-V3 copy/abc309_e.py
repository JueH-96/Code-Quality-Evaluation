import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    p = [0] * (N+1)
    for i in range(2, N+1):
        p[i] = int(data[idx])
        idx += 1
    # Build children list
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        children[p[i]].append(i)
    # Initialize coverage
    coverage = [0] * (N+1)
    for _ in range(M):
        x = int(data[idx])
        y = int(data[idx+1])
        idx += 2
        # BFS to find all descendants within y generations
        q = deque()
        q.append((x, 0))
        while q:
            node, gen = q.popleft()
            if gen > y:
                continue
            coverage[node] = 1
            for child in children[node]:
                q.append((child, gen+1))
    # Count the number of people covered
    print(sum(coverage))

if __name__ == "__main__":
    main()