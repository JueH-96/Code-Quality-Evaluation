import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    p = list(map(int, data[idx:idx+N-1]))
    idx += N-1
    x = []
    y = []
    for _ in range(M):
        x.append(int(data[idx]))
        y.append(int(data[idx+1]))
        idx += 2
    
    # Build the tree
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        children[p[i-2]].append(i)
    
    # Initialize coverage
    coverage = [0] * (N+1)
    
    # Process each insurance
    for i in range(M):
        xi = x[i]
        yi = y[i]
        q = deque()
        q.append((xi, 0))
        while q:
            node, depth = q.popleft()
            if depth > yi:
                continue
            coverage[node] = 1
            for child in children[node]:
                q.append((child, depth+1))
    
    # Count the number of covered people
    count = sum(coverage[1:])
    print(count)

if __name__ == "__main__":
    main()