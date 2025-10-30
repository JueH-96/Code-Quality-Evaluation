import sys

def main():
    data = sys.stdin.read().split()
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
    
    base = 10**9
    low_int = 0
    high_int = 10000 * base + 1
    
    for _ in range(70):
        mid_int = (low_int + high_int) // 2
        dp = [-10**30] * (n+1)
        dp[1] = 0
        
        for u in range(1, n+1):
            if dp[u] == -10**30:
                continue
            for edge in graph[u]:
                v, b, c = edge
                w = b * base - mid_int * c
                new_val = dp[u] + w
                if new_val > dp[v]:
                    dp[v] = new_val
        
        if dp[n] >= 0:
            low_int = mid_int
        else:
            high_int = mid_int
    
    ans = low_int / base
    print("{:.15f}".format(ans))

if __name__ == "__main__":
    main()