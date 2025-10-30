def min_cost_to_form_t(t, bags):
    n = len(bags)
    m = len(t)
    
    # Initialize DP table
    INF = float('inf')
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    for i in range(1, n + 1):
        dp[i][0] = 0  # We can always form an empty string by skipping all bags
        for j in range(1, m + 1):
            # Skip the bag
            dp[i][j] = dp[i-1][j]
            
            # Try to use one of the strings from the bag
            for s in bags[i-1]:
                if j >= len(s) and t[j-len(s):j] == s:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-len(s)] + 1)
    
    return dp[n][m] if dp[n][m] != INF else -1

def main():
    t = input().strip()
    n = int(input().strip())
    bags = []
    for _ in range(n):
        line = input().strip().split()
        a_i = int(line[0])
        strings = line[1:a_i+1]
        bags.append(strings)
    
    result = min_cost_to_form_t(t, bags)
    print(result)

if __name__ == "__main__":
    main()