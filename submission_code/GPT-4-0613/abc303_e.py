import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    g = defaultdict(list)
    for _ in range(n - 1):
        u, v = map(int, sys.stdin.readline().split())
        g[u].append(v)
        g[v].append(u)
    stars = sorted(len(g[i]) - 1 for i in g if len(g[i]) > 1)
    print(*stars)

if __name__ == "__main__":
    main()