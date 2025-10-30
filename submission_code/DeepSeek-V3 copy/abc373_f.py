def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    W = int(data[1])
    
    w = []
    v = []
    index = 2
    for _ in range(N):
        w.append(int(data[index]))
        v.append(int(data[index+1]))
        index += 2
    
    # Initialize DP table
    dp = [ -1 ] * (W + 1)
    dp[0] = 0
    
    for i in range(N):
        current_w = w[i]
        current_v = v[i]
        # Iterate in reverse to avoid overwriting
        for j in range(W, -1, -1):
            if dp[j] == -1:
                continue
            # Try adding k items of type i
            # The maximum k is min(10**10, (W - j) // current_w)
            # But since 10**10 is very large, we can limit k to (W - j) // current_w
            max_k = (W - j) // current_w
            # To find the optimal k, we need to maximize k * current_v - k * k
            # The function f(k) = k * current_v - k * k is a quadratic function
            # Its maximum is at k = current_v / 2
            # Since k must be integer, we check k = floor(current_v / 2) and ceil(current_v / 2)
            # But since the function is symmetric around the maximum, we can just take floor(current_v / 2)
            # However, k must be <= max_k
            optimal_k = min(max_k, current_v // 2)
            # Also, k can be 0
            # So we try k from 0 to optimal_k
            # But to optimize, we can directly compute the best k
            # The function f(k) = k * current_v - k * k reaches maximum at k = current_v / 2
            # So the best k is min(max_k, current_v // 2)
            # But we need to check k = min(max_k, current_v // 2) and k = min(max_k, (current_v // 2) + 1)
            # Because the function is symmetric around the maximum
            k1 = min(max_k, current_v // 2)
            k2 = min(max_k, (current_v // 2) + 1)
            # Compute the happiness for k1 and k2
            happiness1 = k1 * current_v - k1 * k1
            happiness2 = k2 * current_v - k2 * k2
            # Choose the maximum
            if happiness1 > happiness2:
                best_k = k1
                best_happiness = happiness1
            else:
                best_k = k2
                best_happiness = happiness2
            # Update the DP table
            new_weight = j + best_k * current_w
            if new_weight <= W:
                if dp[new_weight] < dp[j] + best_happiness:
                    dp[new_weight] = dp[j] + best_happiness
    
    # Find the maximum value in the DP table
    max_happiness = max(dp)
    print(max_happiness)

if __name__ == "__main__":
    main()