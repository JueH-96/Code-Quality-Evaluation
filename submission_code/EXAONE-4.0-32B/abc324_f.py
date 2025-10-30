import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    graph = [[] for _ in range(n+1)]
    
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        b = int(next(it))
        c = int(next(it))
        graph[u].append((v, b, c))
    
    lo, hi = 0.0, 10000.0
    for _ in range(70):
        mid = (lo + hi) / 2.0
        dp = [-10**18] * (n+1)
        dp[1] = 0.0
        
        for u in range(1, n+1):
            current = dp[u]
            for (v, b, c) in graph[u]:
                weight = b - mid * c
                new_val = current + weight
                if new_val > dp[v]:
                    dp[v] = new_val
        
        if dp[n] >= 0.0:
            lo = mid
        else:
            hi = mid
            
    print("{:.15f}".format(lo))

if __name__ == "__main__":
    main()