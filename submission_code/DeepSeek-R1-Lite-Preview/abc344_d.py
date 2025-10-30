def main():
    import sys
    data = sys.stdin.read().splitlines()
    T = data[0]
    N = int(data[1])
    bags = []
    for i in range(2, 2+N):
        parts = data[i].split()
        A_i = int(parts[0])
        strings = parts[1:1+A_i]
        bags.append(strings)
    
    len_T = len(T)
    INF = float('inf')
    # Initialize DP table
    dp = [[INF] * (N + 1) for _ in range(len_T + 1)]
    dp[0][0] = 0
    
    for bag in range(1, N + 1):
        for pos in range(len_T + 1):
            if dp[pos][bag - 1] < INF:
                # Option 1: Do nothing
                dp[pos][bag] = min(dp[pos][bag], dp[pos][bag - 1])
                # Option 2: Choose a string from the current bag
                for s in bags[bag - 1]:
                    len_s = len(s)
                    if pos + len_s <= len_T and T[pos:pos + len_s] == s:
                        dp[pos + len_s][bag] = min(dp[pos + len_s][bag], dp[pos][bag - 1] + 1)
    
    result = dp[len_T][N]
    if result < INF:
        print(result)
    else:
        print(-1)

if __name__ == "__main__":
    main()