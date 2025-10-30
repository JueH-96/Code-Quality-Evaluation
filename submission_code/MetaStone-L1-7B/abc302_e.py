import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1

    adj = defaultdict(set)
    degrees = defaultdict(int)
    isolated = N

    for _ in range(Q):
        query = data[idx]
        idx += 1
        if query == '1':
            u = int(data[idx])
            idx += 1
            v = int(data[idx])
            idx += 1
            if degrees[u] == 0:
                isolated -= 1
            if degrees[v] == 0:
                isolated -= 1
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
        elif query == '2':
            v = int(data[idx])
            idx += 1
            current_edges = list(adj[v])
            for w in current_edges:
                adj[v].remove(w)
                adj[w].remove(v)
                degrees[v] -= 1
                degrees[w] -= 1
                if degrees[v] == 0:
                    isolated += 1
                if degrees[w] == 0:
                    isolated += 1
        print(isolated)

if __name__ == '__main__':
    main()