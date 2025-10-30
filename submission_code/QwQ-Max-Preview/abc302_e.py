import sys
from sys import stdin

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    adj = [set() for _ in range(N+1)]  # 1-based indexing
    isolated_count = N

    for _ in range(Q):
        parts = input[ptr:ptr+ (2 if input[ptr] == '2' else 3)]
        ptr += len(parts)
        if parts[0] == '1':
            u = int(parts[1])
            v = int(parts[2])
            adj[u].add(v)
            adj[v].add(u)
            if len(adj[u]) == 1:
                isolated_count -= 1
            if len(adj[v]) == 1:
                isolated_count -= 1
        else:
            v = int(parts[1])
            neighbors = list(adj[v])
            for u in neighbors:
                adj[v].remove(u)
                adj[u].remove(v)
                if len(adj[u]) == 0:
                    isolated_count += 1
            if len(neighbors) > 0:
                isolated_count += 1
        print(isolated_count)

if __name__ == "__main__":
    main()