import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    
    rev_graph = [[] for _ in range(n+1)]
    
    for _ in range(m):
        l = int(next(it)); d = int(next(it)); k = int(next(it)); c = int(next(it))
        A = int(next(it)); B = int(next(it))
        rev_graph[B].append((l, d, k, c, A))
    
    NEG_INF = -10**30
    dp = [NEG_INF] * (n+1)
    BIG = 10**30
    dp[n] = BIG
    
    heap = [(-BIG, n)]
    
    while heap:
        neg_val, u = heapq.heappop(heap)
        current_val = -neg_val
        if current_val != dp[u]:
            continue
            
        for edge in rev_graph[u]:
            l, d, k, c, w = edge
            if u == n:
                candidate = l + (k-1) * d
            else:
                if dp[u] < c + l:
                    continue
                n_max = (dp[u] - c - l) // d
                if n_max >= k:
                    n_max = k - 1
                candidate = l + n_max * d
                
            if candidate > dp[w]:
                dp[w] = candidate
                heapq.heappush(heap, (-candidate, w))
                
    for i in range(1, n):
        if dp[i] < 0:
            print("Unreachable")
        else:
            print(dp[i])

if __name__ == "__main__":
    main()