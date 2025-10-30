# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, b, c = map(int, input().split())
        edges.append((u, v, b, c))

    def check(x):
        dist = [-float('inf')] * (n + 1)
        dist[1] = 0
        for _ in range(n):
            updated = False
            for u, v, b, c in edges:
                if dist[u] != -float('inf') and dist[v] < dist[u] + b - x * c:
                    dist[v] = dist[u] + b - x * c
                    updated = True
            if not updated:
                break
        return dist[n] >= 0

    left = 0
    right = 10000
    for _ in range(100):
        mid = (left + right) / 2
        if check(mid):
            left = mid
        else:
            right = mid
    
    print(left)

solve()