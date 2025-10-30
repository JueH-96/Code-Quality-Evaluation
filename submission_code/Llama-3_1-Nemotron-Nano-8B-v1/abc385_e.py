import sys
from sys import stdin
def main():
    sys.setrecursionlimit(1 << 25)
    N = int(stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    degree = [0] * (N+1)
    for u in range(1, N+1):
        degree[u] = len(edges[u])
    
    max_kept = 0
    for u in range(1, N+1):
        leaf_children = []
        for v in edges[u]:
            cnt = 0
            for w in edges[v]:
                if w != u and degree[w] == 1:
                    cnt += 1
            leaf_children.append(cnt)
        leaf_children.sort(reverse=True)
        for k in range(1, len(leaf_children)+1):
            y = leaf_children[k-1]
            if y < 1:
                break
            current = 1 + k + k * y
            if current > max_kept:
                max_kept = current
    print(N - max_kept)

if __name__ == '__main__':
    main()