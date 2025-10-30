def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    
    dp = [-1] * (W + 1)
    dp[0] = 0
    
    for _ in range(N):
        w = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        
        # Compute the maximum possible k for this type
        max_k = min((W // w), v // 2)
        best_h = []
        best_k = 0
        max_h = -1
        for k in range(max_k, -1, -1):
            if k * w > W:
                continue
            current_h = k * v - k * k
            if current_h > max_h:
                max_h = current_h
                best_k = k
        
        # Update DP in reverse order
        for j in range(W, -1, -1):
            if dp[j] == -1:
                continue
            new_j = j + best_k * w
            if new_j > W:
                continue
            new_h = dp[j] + max_h
            if new_h > dp[new_j]:
                dp[new_j] = new_h
    
    print(max(dp))

if __name__ == '__main__':
    main()