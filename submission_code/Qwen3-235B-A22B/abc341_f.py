import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(data[idx]) - 1
        idx += 1
        v = int(data[idx]) - 1
        idx += 1
        adj[u].append(v)
        adj[v].append(u)
    
    W = list(map(int, data[idx:idx+N]))
    idx += N
    A = list(map(int, data[idx:idx+N]))
    idx += N
    
    # Sort nodes by increasing W
    nodes = list(range(N))
    nodes.sort(key=lambda x: W[x])
    
    gain = [0] * N
    
    for x in nodes:
        candidates = []
        sum_weights = 0
        sum_gains = 0
        for y in adj[x]:
            if W[y] < W[x]:
                candidates.append((W[y], gain[y]))
                sum_weights += W[y]
                sum_gains += gain[y]
        capacity = W[x] - 1
        
        if sum_weights <= capacity:
            gain[x] = 1 + sum_gains
        else:
            if capacity <= 0 or not candidates:
                gain[x] = 1
                continue
            
            dp = [0] * (capacity + 1)
            for (w, v) in candidates:
                if w > capacity:
                    continue
                for j in range(capacity, w - 1, -1):
                    if dp[j - w] + v > dp[j]:
                        dp[j] = dp[j - w] + v
            max_val = max(dp)
            gain[x] = 1 + max_val
    
    total = 0
    for i in range(N):
        total += A[i] * gain[i]
    print(total)

if __name__ == "__main__":
    main()