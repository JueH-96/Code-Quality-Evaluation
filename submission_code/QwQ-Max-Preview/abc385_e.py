import sys
from sys import stdin

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(stdin.readline())
    if N == 1:
        print(0)
        return
    edges = [[] for _ in range(N + 1)]  # nodes are 1-based
    for _ in range(N - 1):
        u, v = map(int, stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    degree = [0] * (N + 1)
    for u in range(1, N + 1):
        degree[u] = len(edges[u])
    
    leaves_count = [0] * (N + 1)
    for u in range(1, N + 1):
        cnt = 0
        for v in edges[u]:
            if degree[v] == 1:
                cnt += 1
        leaves_count[u] = cnt
    
    max_size = 0
    for p in range(1, N + 1):
        is_leaf = (degree[p] == 1)
        S = []
        for c in edges[p]:
            lc = leaves_count[c] - (1 if is_leaf else 0)
            if lc >= 1:
                S.append(lc)
        S.sort(reverse=True)
        current_max = 0
        for x in range(1, len(S) + 1):
            y = S[x - 1]
            temp = x * (y + 1)
            if temp > current_max:
                current_max = temp
        current_size = 1 + current_max
        if current_size > max_size:
            max_size = current_size
    print(N - max_size)

if __name__ == "__main__":
    main()