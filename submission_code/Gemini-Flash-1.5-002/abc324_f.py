# YOUR CODE HERE
import heapq

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, b, c = map(int, input().split())
        edges.append((u, v, b, c))

    max_ratio = 0.0

    q = [(0, 1, 0, 0)]  # (ratio, node, total_beauty, total_cost)

    while q:
        ratio, u, total_beauty, total_cost = heapq.heappop(q)
        
        if u == n:
            max_ratio = max(max_ratio, total_beauty / total_cost if total_cost >0 else 0)
            continue

        for edge in edges:
            if edge[0] == u:
                new_beauty = total_beauty + edge[2]
                new_cost = total_cost + edge[3]
                new_ratio = new_beauty / new_cost if new_cost > 0 else 0
                heapq.heappush(q, (-new_ratio, edge[1], new_beauty, new_cost))

    print(max_ratio)

solve()