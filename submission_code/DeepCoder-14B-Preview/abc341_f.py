def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    edges = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(data[idx])
        idx += 1
        v = int(data[idx])
        idx += 1
        edges[u].append(v)
        edges[v].append(u)
    
    W_list = list(map(int, data[idx:idx + N]))
    idx += N
    W = [0] + W_list  # W[1] is W_1
    
    A_list = list(map(int, data[idx:idx + N]))
    idx += N
    A = [0] + A_list  # A[1] is A_1
    
    # Sort nodes based on increasing order of W
    nodes = list(range(1, N + 1))
    nodes.sort(key=lambda x: W[x])
    
    contribution = [0] * (N + 1)  # contribution[0] is unused
    
    for x in nodes:
        adj = edges[x]
        items = []
        for y in adj:
            if y == x:
                continue
            items.append((W[y], contribution[y]))
        
        max_sum_weight = W[x] - 1
        if max_sum_weight < 0:
            contribution[x] = 1
            continue
        
        # Initialize DP
        dp = [-float('inf')] * (max_sum_weight + 1)
        dp[0] = 0
        
        for w, c in items:
            for s in range(max_sum_weight, w - 1, -1):
                if dp[s - w] != -float('inf'):
                    if dp[s] < dp[s - w] + c:
                        dp[s] = dp[s - w] + c
        
        # Find the maximum value in dp
        max_val = max([val for val in dp if val != -float('inf')] + [-float('inf')])
        
        if max_val == -float('inf'):
            contribution[x] = 1
        else:
            contribution[x] = 1 + max_val
    
    total = sum(A[x] * contribution[x] for x in range(1, N + 1))
    print(total)

if __name__ == '__main__':
    main()