# YOUR CODE HERE
import sys

def solve():
    n, q = map(int, sys.stdin.readline().split())
    adj = [set() for _ in range(n + 1)]

    for _ in range(q):
        query = list(map(int, sys.stdin.readline().split()))
        if query[0] == 1:
            u, v = query[1], query[2]
            adj[u].add(v)
            adj[v].add(u)
        else:
            v, k = query[1], query[2]
            connected = sorted(list(adj[v]) + [v], reverse=True)
            if k > len(connected):
                print(-1)
            else:
                print(connected[k-1])

solve()