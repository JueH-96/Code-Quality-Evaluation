import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    if N == 1:
        print(0)
        return
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    max_size = 0
    for u in range(1, N + 1):
        ys = []
        for v in adj[u]:
            valid = True
            for w in adj[v]:
                if w == u:
                    continue
                if len(adj[w]) != 1:
                    valid = False
                    break
            if valid:
                y = len(adj[v]) - 1
                if y >= 1:
                    ys.append(y)
        # Calculate frequency
        freq = defaultdict(int)
        for y in ys:
            freq[y] += 1
        for y in freq:
            x = freq[y]
            current = 1 + x * (y + 1)
            if current > max_size:
                max_size = current
    print(N - max_size)

if __name__ == '__main__':
    main()