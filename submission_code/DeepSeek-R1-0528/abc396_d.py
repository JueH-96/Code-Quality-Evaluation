import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    edges = []
    index = 2
    graph = [[] for _ in range(n)]
    for i in range(m):
        u = int(data[index])
        v = int(data[index+1])
        w = int(data[index+2])
        index += 3
        u -= 1
        v -= 1
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    total_masks = 1 << n
    dp = [[set() for _ in range(total_masks)] for __ in range(n)]
    
    start_mask = 1
    dp[0][start_mask].add(0)
    
    for mask in range(total_masks):
        for u in range(n):
            if not dp[u][mask]:
                continue
            for (v, w) in graph[u]:
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                for x in dp[u][mask]:
                    new_x = x ^ w
                    dp[v][new_mask].add(new_x)
    
    dest = n-1
    ans = None
    for mask in range(total_masks):
        if not dp[dest][mask]:
            continue
        for x in dp[dest][mask]:
            if ans is None or x < ans:
                ans = x
                
    if ans is None:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()