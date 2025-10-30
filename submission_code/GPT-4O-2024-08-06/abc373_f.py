# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    W = int(data[1])
    
    items = []
    index = 2
    for _ in range(N):
        w = int(data[index])
        v = int(data[index + 1])
        items.append((w, v))
        index += 2
    
    # Initialize the dp array
    dp = [-float('inf')] * (W + 1)
    dp[0] = 0
    
    # Process each item type
    for w_i, v_i in items:
        # We need to iterate backwards to avoid overwriting results we still need to use
        for current_weight in range(W, -1, -1):
            # Try to add k items of this type
            k = 1
            while k * w_i <= W - current_weight:
                happiness = k * v_i - k * k
                new_weight = current_weight + k * w_i
                dp[new_weight] = max(dp[new_weight], dp[current_weight] + happiness)
                k += 1
    
    # The answer is the maximum happiness achievable with weight <= W
    print(max(dp))

main()