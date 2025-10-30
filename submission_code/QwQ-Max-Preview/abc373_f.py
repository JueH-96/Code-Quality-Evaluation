def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    
    items = []
    for _ in range(N):
        w = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        
        k_max = min(v // 2, W // w)
        if k_max == 0:
            continue
        
        # Add k_max
        weight = k_max * w
        value = k_max * v - k_max * k_max
        items.append((weight, value))
        
        # Add k_max -1 if possible
        if k_max >= 1:
            weight_prev = (k_max - 1) * w
            value_prev = (k_max - 1) * v - (k_max - 1) ** 2
            items.append((weight_prev, value_prev))
    
    # 0-1 knapsack
    dp = [0] * (W + 1)
    for w_item, v_item in items:
        for s in range(W, w_item - 1, -1):
            if dp[s - w_item] + v_item > dp[s]:
                dp[s] = dp[s - w_item] + v_item
    
    print(max(dp))

if __name__ == '__main__':
    main()