def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    T = input_data[0]
    N = int(input_data[1])
    
    bags = []
    idx = 2
    for _ in range(N):
        A = int(input_data[idx])
        idx += 1
        strings = input_data[idx:idx+A]
        idx += A
        bags.append(strings)
    
    # We'll use dynamic programming where:
    # dp[i][j] = minimum cost (yen) to form T[:j] after considering up to bag i (0-based indexing for dp).
    # i in [0..N], j in [0..len(T)]
    # Initially dp[0][0] = 0, meaning we formed the empty string with cost 0.
    # If dp[i][j] is defined, then dp[i+1][j] can be at least dp[i][j].
    # Also, if we can append a string s from bag i (1-based in problem, 0-based in code),
    #   we check if T[j:j+len(s)] == s; if yes, dp[i+1][j+len(s)] = min(dp[i+1][j+len(s)], dp[i][j] + 1).
    
    INF = float('inf')
    lenT = len(T)
    
    dp = [[INF]*(lenT+1) for _ in range(N+1)]
    dp[0][0] = 0  # cost 0 to form empty prefix
    
    for i in range(N):
        for j in range(lenT+1):
            if dp[i][j] == INF:
                continue
            
            # Option 1: Do nothing
            dp[i+1][j] = min(dp[i+1][j], dp[i][j])
            
            # Option 2: Pay 1 yen and choose one string from bag i
            for s in bags[i]:
                new_len = j + len(s)
                # We only proceed if new_len <= lenT and T[j:j+len(s)] == s
                if new_len <= lenT and T[j:new_len] == s:
                    dp[i+1][new_len] = min(dp[i+1][new_len], dp[i][j] + 1)
    
    ans = dp[N][lenT]
    if ans == INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()