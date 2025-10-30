import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N, M = int(input[ptr]), int(input[ptr+1])
    ptr += 2
    parents = list(map(int, input[ptr:ptr+(N-1)]))
    ptr += (N-1)
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        p = parents[i-2]
        children[p].append(i)
    max_insurance = [-1] * (N+1)
    for _ in range(M):
        x = int(input[ptr])
        y = int(input[ptr+1])
        ptr += 2
        if max_insurance[x] < y:
            max_insurance[x] = y
    coverage = [-1] * (N+1)
    q = deque()
    q.append(1)
    coverage[1] = max_insurance[1]
    while q:
        u = q.popleft()
        for v in children[u]:
            new_cov = coverage[u] - 1
            candidate = max(new_cov, max_insurance[v])
            coverage[v] = candidate
            q.append(v)
    count = 0
    for i in range(1, N+1):
        if coverage[i] >= 0:
            count += 1
    print(count)

if __name__ == "__main__":
    main()