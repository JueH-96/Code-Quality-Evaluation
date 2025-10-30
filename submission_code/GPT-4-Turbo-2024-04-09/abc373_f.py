def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    W = int(data[1])
    
    items = []
    for i in range(N):
        w = int(data[2 + 2 * i])
        v = int(data[3 + 2 * i])
        items.append((w, v))
    
    # dp[j] will be the maximum happiness we can achieve with exactly j weight
    dp = [-float('inf')] * (W + 1)
    dp[0] = 0  # Base case: 0 weight, 0 happiness
    
    for w, v in items:
        # We need to iterate backwards to prevent using the same item type multiple times in one iteration
        for current_weight in range(W, w - 1, -1):
            # Calculate the maximum k such that k * w <= current_weight
            max_k = current_weight // w
            
            # Calculate the happiness for each possible k from 1 to max_k
            for k in range(1, max_k + 1):
                new_weight = current_weight - k * w
                if new_weight >= 0:
                    happiness_gain = k * v - k * k
                    if dp[new_weight] != -float('inf'):
                        dp[current_weight] = max(dp[current_weight], dp[new_weight] + happiness_gain)
    
    # The answer is the maximum value in dp that is not -inf
    print(max(dp))

if __name__ == "__main__":
    main()