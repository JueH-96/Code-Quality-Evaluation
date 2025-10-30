import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    T0 = 10**25
    dp = [-10**30] * (n + 1)
    dp[n] = T0

    rev_graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        l = int(next(it))
        d = int(next(it))
        k = int(next(it))
        c = int(next(it))
        A = int(next(it))
        B = int(next(it))
        rev_graph[B].append((l, d, k, c, A))
    
    heap = []
    heapq.heappush(heap, (-T0, n))
    
    while heap:
        neg_val, v = heapq.heappop(heap)
        cur_val = -neg_val
        if cur_val != dp[v]:
            continue
        for edge in rev_graph[v]:
            l, d, k, c, A = edge
            max_departure = dp[v] - c
            if max_departure < l:
                continue
            x = (max_departure - l) // d
            if x < 0:
                continue
            if x >= k:
                x = k - 1
            candidate = l + x * d
            if candidate > dp[A]:
                dp[A] = candidate
                heapq.heappush(heap, (-candidate, A))
    
    for i in range(1, n):
        if dp[i] < 0:
            print("Unreachable")
        else:
            print(dp[i])

if __name__ == "__main__":
    main()