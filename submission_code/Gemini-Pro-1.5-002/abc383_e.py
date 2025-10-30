import itertools

def solve():
    n, m, k = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def get_f(x, y, max_weight):
        q = [(x, [])]
        visited = set()
        while q:
            curr, path = q.pop(0)
            if curr == y:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            for u, v, w in edges:
                if w <= max_weight:
                    neighbor = None
                    if u == curr:
                        neighbor = v
                    elif v == curr:
                        neighbor = u
                    if neighbor is not None and neighbor not in visited:
                        q.append((neighbor, path + [(curr, neighbor, w)]))
        return False

    def calculate_f(x, y):
        low = 0
        high = 10**9 + 1
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if get_f(x, y, mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans

    min_sum_f = float('inf')
    for permuted_b in itertools.permutations(b):
        current_sum_f = 0
        for i in range(k):
            current_sum_f += calculate_f(a[i], permuted_b[i])
        min_sum_f = min(min_sum_f, current_sum_f)

    print(min_sum_f)

solve()