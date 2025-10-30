def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    C = list(map(int, data[2:2+N]))
    
    # Precompute the cost for flipping each character
    flip_cost = [0] * N
    for i in range(N):
        if S[i] == '0':
            flip_cost[i] = C[i]
        else:
            flip_cost[i] = 0
    
    # Precompute the cost for not flipping each character
    no_flip_cost = [0] * N
    for i in range(N):
        if S[i] == '1':
            no_flip_cost[i] = C[i]
        else:
            no_flip_cost[i] = 0
    
    # Initialize DP arrays
    # dp[i][state] where state is 0 or 1 representing the current character
    # and the number of consecutive pairs so far
    # We need to track the number of consecutive pairs, but since we can have at most one, we can manage it
    # with a flag indicating whether we have already had a consecutive pair
    
    # Initialize DP with infinity
    INF = float('inf')
    dp = [[INF] * 2 for _ in range(N+1)]
    dp[0][0] = 0
    
    for i in range(N):
        current_char = S[i]
        for prev_state in range(2):
            if dp[i][prev_state] == INF:
                continue
            # Option 1: Flip the current character
            new_char = '1' if current_char == '0' else '0'
            cost = flip_cost[i]
            # Determine if flipping creates a consecutive pair
            if i > 0:
                prev_char = '1' if S[i-1] == '0' else '0' if dp[i-1][prev_state] != INF else S[i-1]
                if prev_char == new_char:
                    if prev_state == 1:
                        # Already had a consecutive pair, cannot have another
                        continue
                    else:
                        new_state = 1
                else:
                    new_state = prev_state
            else:
                new_state = prev_state
            dp[i+1][new_state] = min(dp[i+1][new_state], dp[i][prev_state] + cost)
            # Option 2: Do not flip the current character
            new_char = current_char
            cost = no_flip_cost[i]
            if i > 0:
                prev_char = S[i-1] if dp[i-1][prev_state] != INF else S[i-1]
                if prev_char == new_char:
                    if prev_state == 1:
                        # Already had a consecutive pair, cannot have another
                        continue
                    else:
                        new_state = 1
                else:
                    new_state = prev_state
            else:
                new_state = prev_state
            dp[i+1][new_state] = min(dp[i+1][new_state], dp[i][prev_state] + cost)
    
    # The final state must have exactly one consecutive pair
    result = dp[N][1]
    print(result)

if __name__ == "__main__":
    main()