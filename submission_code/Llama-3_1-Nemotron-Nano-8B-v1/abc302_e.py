import sys
from sys import stdin

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1

    adj = [set() for _ in range(N + 1)]
    degree = [0] * (N + 1)
    count = N

    for _ in range(Q):
        query = input[idx]
        idx += 1
        if query == '1':
            u = int(input[idx])
            idx += 1
            v = int(input[idx])
            idx += 1
            # Process type 1: add edge u-v
            for x in [u, v]:
                if degree[x] == 0:
                    count -= 1
                degree[x] += 1
            adj[u].add(v)
            adj[v].add(u)
        else:
            v = int(input[idx])
            idx += 1
            # Process type 2: remove all edges connected to v
            edges = list(adj[v])
            for u in edges:
                adj[u].remove(v)
            adj[v].clear()
            prev_degree_v = degree[v]
            if prev_degree_v > 0:
                count += 1
            degree[v] = 0
            temp = 0
            for u in edges:
                if degree[u] == 1:
                    temp += 1
                degree[u] -= 1
            count += temp
        print(count)

if __name__ == "__main__":
    main()