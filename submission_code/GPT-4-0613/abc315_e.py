import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    G = [[] for _ in range(N+1)]
    indeg = [0]*(N+1)
    for i in range(1, N+1):
        line = list(map(int, sys.stdin.readline().split()))
        C = line[0]
        for j in range(1, C+1):
            G[line[j]].append(i)
            indeg[i] += 1
    que = deque([i for i in range(1, N+1) if indeg[i] == 0])
    ans = []
    while que:
        v = que.popleft()
        if v == 1:
            break
        ans.append(v)
        for nv in G[v]:
            indeg[nv] -= 1
            if indeg[nv] == 0:
                que.append(nv)
    print(*ans[::-1])

if __name__ == "__main__":
    main()