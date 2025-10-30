import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    m = int(data[1])
    graph = [[] for _ in range(n)]
    index = 2
    
    for _ in range(m):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        index += 2
        graph[u].append(v)
        graph[v].append(u)
    
    weights = list(map(int, data[index:index + n]))
    index += n
    A = list(map(int, data[index:index + n]))
    
    sorted_vertices = sorted(range(n), key=lambda i: weights[i])
    f = [0] * n
    
    for i in range(n):
        x = sorted_vertices[i]
        cap = weights[x] - 1
        if cap < 0:
            f[x] = 1
            continue
        
        dp = [0] * (cap + 1)
        
        for y in graph[x]:
            if weights[y] < weights[x]:
                w_y = weights[y]
                val_y = f[y]
                if w_y > cap:
                    continue
                for j in range(cap, w_y - 1, -1):
                    if j >= w_y and dp[j] < dp[j - w_y] + val_y:
                        dp[j] = dp[j - w_y] + val_y
        
        best_val = max(dp) if cap >= 0 else 0
        f[x] = 1 + best_val
    
    total_ops = 0
    for i in range(n):
        total_ops += A[i] * f[i]
    
    print(total_ops)

if __name__ == "__main__":
    main()