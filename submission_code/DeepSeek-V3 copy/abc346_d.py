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
    # dp[i][j][k]: i-th character, j is the current character (0 or 1), k is the number of consecutive pairs (0 or 1)
    # Initialize with infinity
    INF = float('inf')
    dp = [[[INF for _ in range(2)] for _ in range(2)] for _ in range(N+1)]
    
    # Base case: first character
    # If we flip the first character
    dp[1][1][0] = flip_cost[0]
    # If we don't flip the first character
    dp[1][0][0] = no_flip_cost[0]
    
    # Fill the DP table
    for i in range(1, N):
        for prev_char in range(2):
            for consecutive in range(2):
                if dp[i][prev_char][consecutive] == INF:
                    continue
                # Current character can be 0 or 1
                for curr_char in range(2):
                    # Calculate the new consecutive count
                    if curr_char == prev_char:
                        new_consecutive = consecutive + 1
                    else:
                        new_consecutive = 0
                    # Check if the new_consecutive is valid (<= 1)
                    if new_consecutive > 1:
                        continue
                    # Calculate the cost
                    if curr_char == 0:
                        cost = no_flip_cost[i]
                    else:
                        cost = flip_cost[i]
                    # Update the DP table
                    if dp[i+1][curr_char][new_consecutive] > dp[i][prev_char][consecutive] + cost:
                        dp[i+1][curr_char][new_consecutive] = dp[i][prev_char][consecutive] + cost
    
    # Find the minimum cost among all possible states with exactly one consecutive pair
    min_cost = INF
    for last_char in range(2):
        if dp[N][last_char][1] < min_cost:
            min_cost = dp[N][last_char][1]
    
    print(min_cost)

if __name__ == "__main__":
    main()