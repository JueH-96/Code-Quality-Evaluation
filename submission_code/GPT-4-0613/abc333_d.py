import sys
from collections import defaultdict
input = sys.stdin.readline

def main():
    N = int(input())
    edges = defaultdict(int)
    for _ in range(N-1):
        u, v = map(int, input().split())
        edges[u] += 1
        edges[v] += 1
    if edges[1] == 1:
        print(1)
    else:
        print(sum(v == 1 for v in edges.values()) // 2 + 1)

if __name__ == "__main__":
    main()