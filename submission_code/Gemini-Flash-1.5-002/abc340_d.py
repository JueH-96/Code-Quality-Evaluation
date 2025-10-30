# YOUR CODE HERE
import sys

def solve():
    n = int(sys.stdin.readline())
    edges = []
    for _ in range(n - 1):
        a, b, x = map(int, sys.stdin.readline().split())
        edges.append((a, b, x -1))

    dist = [float('inf')] * n
    dist[0] = 0

    for i in range(n):
        for j in range(n - 1):
            a, b, x = edges[j]
            if dist[j] != float('inf'):
                dist[j+1] = min(dist[j+1], dist[j] + a)
                dist[x] = min(dist[x], dist[j] + b)

    print(dist[n-1])

solve()