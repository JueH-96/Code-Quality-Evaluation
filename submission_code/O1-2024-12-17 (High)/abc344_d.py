def main():
    import sys
    input = sys.stdin.readline

    # Read inputs
    T = input().strip()
    N = int(input().strip())
    
    bags = []
    for _ in range(N):
        line = input().split()
        A_i = int(line[0])
        bag_strings = line[1:]
        bags.append(bag_strings)
    
    # Set up DP array
    lenT = len(T)
    INF = 10**9
    dp = [[INF]*(lenT+1) for _ in range(N+1)]
    
    # Base case: cost 0 to form empty string with 0 steps
    dp[0][0] = 0
    
    # Fill DP
    for i in range(1, N+1):
        for pref_len in range(lenT+1):
            if dp[i-1][pref_len] == INF:
                continue
            
            # Option 1: Do nothing
            dp[i][pref_len] = min(dp[i][pref_len], dp[i-1][pref_len])
            
            # Option 2: Pay 1 yen and choose a string from bag i
            for s in bags[i-1]:
                new_len = pref_len + len(s)
                # Check if chosen string s matches T from pref_len
                if new_len <= lenT and T[pref_len:new_len] == s:
                    dp[i][new_len] = min(dp[i][new_len], dp[i-1][pref_len] + 1)
    
    # Get the result
    answer = dp[N][lenT]
    print(answer if answer != INF else -1)

# Do not forget to call main()
if __name__ == "__main__":
    main()