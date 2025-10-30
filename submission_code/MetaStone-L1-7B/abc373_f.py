def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    W = int(data[1])
    index = 2
    
    group_options = []
    for _ in range(N):
        v = int(data[index])
        w = int(data[index + 1])
        index += 2
        
        max_k = min(v // 2, W // w)
        for k in range(0, max_k + 1):
            w_total = k * w
            v_total = k * v - k * k
            group_options.append((w_total, v_total))
    
    dp = [-float('inf')] * (W + 1)
    dp[0] = 0
    
    for w, v in group_options:
        for j in range(W, w - 1, -1):
            if dp[j - w] != -float('inf'):
                if dp[j] < dp[j - w] + v:
                    dp[j] = dp[j - w] + v
    
    max_happiness = max(dp)
    print(max_happiness if max_happiness != -float('inf') else 0)

if __name__ == '__main__':
    main()