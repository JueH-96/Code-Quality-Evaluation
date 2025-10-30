import heapq
import itertools

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def calculate_f(x, y, max_weight):
        dist = {i: float('inf') for i in range(1, n + 1)}
        dist[x] = 0
        pq = [(0, x)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue

            for v1, v2, w in edges:
                v = None
                if v1 == u:
                    v = v2
                elif v2 == u:
                    v = v1
                else:
                    continue

                if w <= max_weight:
                    if dist[v] > dist[u] + w:
                        dist[v] = dist[u] + w
                        heapq.heappush(pq, (dist[v], v))
        
        if dist[y] == float('inf'):
            return float('inf')
        else:
            return dist[y] != float('inf')

    def find_min_path_weight(x, y):
        low = 0
        high = 10**9
        ans = float('inf')

        while low <= high:
            mid = (low + high) // 2
            if calculate_f(x, y, mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans
    
    min_sum = float('inf')
    for perm in itertools.permutations(b):
        current_sum = 0
        for i in range(k):
            current_sum += find_min_path_weight(a[i], perm[i])
        min_sum = min(min_sum, current_sum)
    
    print(min_sum)

solve()